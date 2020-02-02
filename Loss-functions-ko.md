## Loss Function이란 무엇인가요? ##

Loss function은 모델이 얼마나 정확하게 예측하는지에 대한 피드백을 네트워크에 제공함으로써, 딥러닝 알고리즘을 학습할 때 가장 중요한 구성 요소 중 하나입니다. "Loss function"은 일반적으로 ground truth를 포함한 하나의 학습 배치(batch)에 대해 네트워크에서 출력된 모든 값들을 loss(또는 예측의 부정확함)로 표현하는 단일 값에 매핑하는 수학 함수입니다. 
Loss function의 값이 높을수록 일반적으로 알고리즘의 출력 예측이 실제와 크게 다르다는 것을 의미합니다.
Cross Entropy loss, Mean Squared Error 및 그 변형들이 가장 일반적으로 사용되는 loss function입니다.
CNN, RNN 또는 LSTM과 같은 일반적인 딥러닝 네트워크에서는 학습이 진행됨에 따라 loss 값이 감소할 것으로 예상되며, 이는 모델이 올바르게 학습하고 있음을 의미합니다.

## GAN의 loss function은 어떻게 다른가요? ##

GAN의 loss function은 위에서 설명한 일반 네트워크의 loss function과 다르지 않습니다. 그러나, GAN의 가장 중요한 속성은 2개의 네트워크가 별도로(또는 교대로) generator와 critic(discriminator)로 학습해야 한다는 것입니다. 결과적으로, 이들 네트워크 각각은 각각의 학습 동안 자신의 loss function을 가지게 됩니다.

*__Critic loss function:__*: Critic 학습은 고양이와 개 클래스가 "실제"와 "가짜" 클래스로 대체된다는 점을 제외하고 고양이-개 감지기와 같은 모든 표준 알고리즘을 학습하는 것과 매우 유사합니다. Critic loss function은 실제 데이터가 "실제"로 분류되고 generator의 가짜 데이터가 한 번의 critic 학습에 대해 "가짜"로 얼마나 정확하게 분류되는지 평가합니다. Critic loss function은 generator의 "가짜" 데이터가 종종 critic에 의해 "실제"로 분류되어 generator에 의해 쉽게 속이는 것을 암시하는 경우 큰 값을 출력합니다.

*__Generator loss function:__*: Generator 학습 중에 생성된 데이터가 이를 속일 만큼 현실적인지 확인하기 위해, 하나의 배치에 대한 generator의 출력이 critic에게 공급(feed into)됩니다. 잘 학습된 generator는 일반적으로 critic에게 생성된 데이터가 실제라고 믿게 하여 loss function score가 더 작아질 것이라고 속이는 것입니다.

## GAN loss function들의 문제점은 무엇인가요?? ##

Cross entropy와 같은 표준 loss function을 사용하는 경우, GAN loss function에서 많은 문제점들이 발생합니다.

*__(1) Oscillating losses:__* 이전 섹션에서 언급했듯이 표준 고양이-개 감지기 유형 문제에서 학습이 진행됨에 따라 loss function 값이 감소할 것으로 예상합니다. 그러나 이것은 GAN에게는 해당되지 않습니다. Generator와 critic이 교대로 학습을 수행하기 때문에, 그들은 서로에 대해 "더 강하고" 더 "약하게" loss를 얻음으로써 loss값들의 진동이 일어납니다(oscillating). 이것은 학습이 전반적으로 잘 진행되었는지 여부와 generator에 의해 생성된 데이터가 충분한지 이해하기 어렵게 만듭니다.

*__(2) Mode collapse:__* 종종, generator는 critic이 항상 속아넘거가데 되는 1개의 특정 출력에 대해 critic을 속이는 것을 학습합니다. 이렇게 되면 generator는 새로운 종류의 데이터를 학습하고 생성하려는 동기가 없어져 최적의 generator 되지 못한다.

## Wasserstein loss function과 WGAN은 무엇인가요? ##

위에서 논의된 문제는 Wasserstein loss라는 새로운 유형의 loss function로 극복되었으며, 이를 사용하는 GAN을 WGAN(Wasserstein GAN) 이라고 합니다. 실무에 적용 시에는 수학적으로 깊은 이해 없이,
Wasserstein loss가 주로 위에서 논의한 문제를 해결한다는 것을 아는 것으로 충분합니다.

(1) Loss 값과 모델 학습 간에 상관 관계가 있도록 Loss function을 정의합니다.

(2) GAN이 시간이 지남에 따라 안정적이 되도록 합니다.

## Loss function은 DeepComposer에서 어떻게 작동하나요? ##

GAN loss function 대해 언급한 모든 내용은 DeepComposer에도 적용되므로, DeepComposer 아키텍처에서도 Wasserstein loss function을 사용합니다. 
그러나 Wasserstein loss function은 GAN에 잘 맞는 또 다른 유형의 loss function에 불과하기 때문에, DeepComposer에서 사용되는 loss function에 대해 명확히 이해하는 것이 중요합니다. 
이를 염두에 두고, loss function이 DeepComposer 아키텍처 어떻게 관련되어 있는지 확인해 봅시다.

*__(1) Critic loss function:__* Critic 학습 중에는 실제 데이터(멜로디가 있는 멀티 트랙 곡 + 학습 데이터셋의 반주)와 가짜 데이터(generator에 의해 생성된 악기 반주들에서 생성된 노래)의 2가지 타입 데이터가 필요합니다.
Loss function은 critic이 실제 노래를 "실제"로 식별하고 가짜 노래를 "가짜"로 얼마나 잘 식별할 수 있는지를 나타내는 값을 출력합니다. 값이 높으면 critic이 약하고 generator에 의해 쉽게 속이는 것을 나타내므로, 이 값은 critic의 학습을 개선하기 위한 피드백을 제공하는 데 사용됩니다.

*__(2) Generator loss function:__* Generator 학습 중에 generator는 단일 멜로디 트랙을 입력으로 받아 멀티 트랙 곡을 출력합니다. 이 결과는 이제 critic(weight들이 동결된)에게 전달되어 critic이 실제 곡들의 분포 대비 얼마나 가깝게 생각하는지 알아봅니다. 높은 값을 산출하는 loss function은 critic이 올바르게 속지 않았음을 의미하므로, generator가 더 잘 학습하도록 페널티를 부여합니다.