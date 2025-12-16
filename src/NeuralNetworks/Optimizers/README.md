# Optimizers

## Stochastic Gradient Descent (SGD)

## Stochastic Gradient Descent with Momentum

File: [`StochasticGradientDescentMomentum.cs`](StochasticGradientDescentMomentum.cs)

### Description

> With Stochastic Gradient Descent we don’t compute the exact derivate of our loss function. Instead, we’re estimating it on a small batch. Which means we’re not always going in the optimal direction, because our derivatives are ‘noisy’. So, exponentially weighed averages can provide us a better estimate which is closer to the actual derivate than our noisy calculations. This is one reason why momentum might work better than classic SGD.

> The other reason lies in ravines. Ravine is an area, where the surface curves much more steeply in one dimension than in another. Ravines are common near local minimas in deep learning and SGD has troubles navigating them. SGD will tend to oscillate across the narrow ravine since the negative gradient will point down one of the steep sides rather than along the ravine towards the optimum. Momentum helps accelerate gradients in the right direction.[^tds]

[^tds] https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d

Adam wg AI:

Optymalizator Adam (Adaptive Moment Estimation) to popularny i efektywny algorytm u¿ywany w g³êbokim uczeniu do trenowania sieci neuronowych, który adaptacyjnie dostosowuje tempo uczenia siê dla ka¿dego parametru, ³¹cz¹c zalety Momentum (wykorzystuje œredni¹ historycznych gradientów) i RMSprop (skaluje tempo uczenia na podstawie œredniej kwadratów gradientów). Dzia³a iteracyjnie, pomagaj¹c modelowi inteligentnie znaleŸæ minimum funkcji straty, szybko zbiegaj¹c i radz¹c sobie z rzadkimi danymi i szumem, przez co jest czêsto domyœlnym wyborem. 

Kluczowe cechy Adam: Adaptacyjne tempo uczenia: Automatycznie dostosowuje szybkoœæ uczenia (learning rate) dla ka¿dego parametru, co prowadzi do szybszej konwergencji ni¿ tradycyjny SGD.Po³¹czenie Momentum i RMSprop: Wykorzystuje "pêd", aby przyspieszyæ w po¿¹danych kierunkach, oraz adaptacyjne skalowanie, aby stabilizowaæ kroki wzd³u¿ gradientów.Efektywnoœæ: Skuteczny w przypadku du¿ych zbiorów danych i z³o¿onych modeli, radzi sobie z rzadkimi danymi (np. w NLP).Parametry: G³ównymi hiperparametrami s¹ tempo uczenia (\(\alpha \)), szybkoœci zaniku dla pierwszego momentu (\(\beta _{1}\)) i drugiego momentu (\(\beta _{2}\)) oraz ma³a sta³a (\(\varepsilon \)) zapobiegaj¹ca dzieleniu przez zero. Jak dzia³a w skrócie: Oblicza pierwszy moment (œredni¹) gradientów (pêd).Oblicza drugi moment (œredni¹ kwadratów) gradientów (informacja o zmiennoœci).Normalizuje kroki, u¿ywaj¹c tych dwóch momentów do dostosowania tempa uczenia siê dla ka¿dego parametru.Wykonuje aktualizacjê wag modelu. Zastosowanie: Jest szeroko stosowany w g³êbokim uczeniu (deep learning) do trenowania sieci neuronowych, od prostych zadañ po z³o¿one modele przetwarzania jêzyka naturalnego (NLP) czy systemy rekomendacji. 