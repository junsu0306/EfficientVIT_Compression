"""
Implements the knowledge distillation loss, proposed in deit
DeiT(Data-efficient Image Transformers)에서 제안된 지식 증류(Knowledge Distillation) 손실 함수 구현.
지식 증류란 학습이 완료된 큰 모델(Teacher)의 지식을 작은 모델(Student)에 전달하는 기법이다.
"""
import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.

    [한국어 설명]
    역할:
        표준 분류 손실(예: CrossEntropy)에 지식 증류 손실을 결합하는 래퍼(wrapper) 모듈.
        Teacher 모델의 예측을 추가적인 감독 신호로 활용하여 Student 모델의 학습을 강화한다.

    핵심 동작:
        1. Student 모델의 출력과 실제 정답 레이블 사이의 기본 손실(base_loss) 계산
        2. Teacher 모델의 예측과 Student의 증류 토큰(dist_token) 출력 사이의 증류 손실 계산
        3. alpha 가중치로 두 손실을 결합: loss = base_loss * (1 - alpha) + distill_loss * alpha

    입력:
        base_criterion: 기본 분류 손실 함수 (예: CrossEntropyLoss)
        teacher_model:  사전 학습된 Teacher 모델 (학습하지 않고 추론만 수행)
        distillation_type: 증류 방식 ('none' / 'soft' / 'hard')
        alpha: 증류 손실의 가중치 (0이면 증류 비활성화, 1이면 증류만 사용)
        tau: Soft KD에서 사용하는 온도 파라미터 (Temperature Scaling)

    출력:
        결합된 최종 손실 스칼라 텐서
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        # 기본 분류 손실 함수 저장 (예: CrossEntropyLoss, LabelSmoothingCrossEntropy)
        self.base_criterion = base_criterion
        # Teacher 모델 저장 - forward 중에는 no_grad로 추론만 수행함
        self.teacher_model = teacher_model
        # 증류 방식은 반드시 세 가지 중 하나여야 함
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        # alpha: 증류 손실 가중치 (0~1 사이 값)
        self.alpha = alpha
        # tau: Soft KD의 온도 파라미터. 값이 클수록 Teacher의 소프트 레이블 분포가 완만해져
        #      Student가 클래스 간 유사도 정보를 더 잘 학습할 수 있음
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion

        [한국어 설명]
        파라미터:
            inputs  : 원본 입력 이미지 텐서 - Teacher 모델에 그대로 전달됨
            outputs : Student 모델의 출력.
                      - 단순 Tensor인 경우: 분류 토큰(class token)의 로짓만 있음
                      - Tuple[Tensor, Tensor]인 경우:
                            outputs[0] = class token 로짓 (base loss 계산용)
                            outputs[1] = distillation token 로짓 (증류 loss 계산용)
                        DeiT 구조에서 cls_token과 dist_token 두 가지 출력이 나옴
            labels  : 실제 정답 클래스 인덱스 (base loss 계산에 사용)

        반환값:
            최종 손실 스칼라 (base_loss와 distillation_loss의 가중 합계)
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            # 모델이 튜플을 반환하는 경우: 첫 번째는 cls_token 출력, 두 번째는 dist_token 출력
            outputs, outputs_kd = outputs

        # 기본 분류 손실: Student의 cls_token 출력과 실제 레이블 사이의 손실
        base_loss = self.base_criterion(outputs, labels)

        # 증류 유형이 'none'이면 기본 손실만 반환 (증류 비활성화)
        if self.distillation_type == 'none':
            return base_loss

        # 증류가 활성화된 경우, dist_token 출력이 반드시 존재해야 함
        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")

        # don't backprop throught the teacher
        # Teacher 모델은 평가 모드로만 사용 - 역전파 없이 추론만 수행하여 메모리/연산 절약
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            # [Soft Knowledge Distillation]
            # Teacher의 소프트 레이블(확률 분포)을 Student가 모방하도록 학습
            #
            # 핵심 수식:
            #   L_soft = T^2 * KL_div( softmax(z_s / T) || softmax(z_t / T) )
            #
            # 여기서:
            #   z_s = Student의 dist_token 로짓 (outputs_kd)
            #   z_t = Teacher의 로짓 (teacher_outputs)
            #   T   = 온도 파라미터 (tau)
            #
            # 온도 T로 나누는 이유:
            #   소프트맥스 출력을 더 완만하게(soft) 만들어 클래스 간 상대적 유사도 정보를 보존.
            #   T=1이면 원래 분포, T>1이면 더 균일한 분포로 "dark knowledge" 전달.
            #
            # T^2를 곱하는 이유:
            #   역전파 시 그래디언트가 1/T^2 배로 축소되는 것을 보정하기 위함.
            #   (Hinton et al., 2015 "Distilling the Knowledge in a Neural Network" 참고)
            #
            # log_target=True를 사용하는 이유:
            #   수치 안정성을 위해 KL_div 계산에서 두 분포 모두 log 공간에서 처리.
            #   F.kl_div(log_p, log_q, log_target=True) = sum(exp(log_q) * (log_q - log_p))
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),    # Student의 로그 소프트맥스 (온도 적용)
                F.log_softmax(teacher_outputs / T, dim=1), # Teacher의 로그 소프트맥스 (온도 적용)
                reduction='sum',    # 배치 전체 합산 후 numel()로 나눠 평균화
                log_target=True     # 두 번째 인자도 log 공간 입력임을 명시
            ) * (T * T) / outputs_kd.numel()
            # outputs_kd.numel() = batch_size * num_classes
            # reduction='sum' + 수동 나눗셈으로 배치 평균화 (F.kl_div reduction='batchmean'과 동일 효과)

        elif self.distillation_type == 'hard':
            # [Hard Knowledge Distillation]
            # Teacher의 예측 클래스(argmax)를 정답 레이블처럼 사용하여 CrossEntropy 계산
            #
            # 핵심 수식:
            #   L_hard = CrossEntropy( z_s, argmax(z_t) )
            #
            # Soft KD와 비교:
            #   - Hard KD: Teacher의 가장 높은 확률 클래스만 사용 (원-핫 레이블처럼 처리)
            #   - Soft KD: Teacher의 전체 확률 분포를 활용 (더 많은 정보 전달)
            #
            # teacher_outputs.argmax(dim=1): Teacher가 가장 높은 확률을 부여한 클래스 인덱스
            distillation_loss = F.cross_entropy(
                outputs_kd, teacher_outputs.argmax(dim=1))

        # [최종 손실 결합]
        # loss = (1 - alpha) * base_loss + alpha * distillation_loss
        # alpha = 0: 증류 손실 완전 무시, 기본 손실만 사용
        # alpha = 1: 기본 손실 무시, 증류 손실만 사용
        # alpha = 0.5: 두 손실을 동등하게 사용
        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss
