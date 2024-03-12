# WIMU projekt
## Design proposal
### Temat projektu
***Symbolic music generation using Mamba architecture***

Przez ostatnie lata wiele wysiłku zostało włożone w to, aby uczynić Transformery coraz bardziej wydajnymi. Jest natomiast możliwe, że będą one stopniowo zastępowane zaproponowaną niedawno, nową architekturą Mamba. Nie znaleźliśmy żadnych badań dotyczących wykorzystania tej architektury do generowania muzyki. W związku z tym chcielibyśmy sami sprawdzić jak Mamba sprawdza się w tym zadaniu.

Projekt ma na celu zbadanie możliwości generowania muzyki symbolicznej przy użyciu architektury Mamba. Docelowo planujemy generować muzykę w formacie MIDI, lecz jest możliwe że ostatecznie skupimy się na formacie ABC lub MusicXML.

### Harmonogram
**UWAGA:** Harmonogram będzie zmieniany dynamicznie wraz z postępem prac.

| Tydzień         | TODO |
|-----------------|------|
| 19 Feb - 25 Feb |==początek semestru==|
| 26 Feb - 03 Mar |wybór tematu|
| 04 Mar - 10 Mar |konsultacja tematu|
| 11 Mar - 17 Mar |design proposal|
| 18 Mar - 24 Mar |przygotowanie repo, MLOps|
| 25 Mar - 31 Mar |==święta==|
| 01 Apr - 07 Apr |zaznajomienie się z architekturą i rozplanowanie implementacji|
| 08 Apr - 14 Apr |rozpoczęcie implementacji|
| 15 Apr - 21 Apr |wybranie formatu danych, datasetu i sposobu tokenizacji|
| 22 Apr - 28 Apr |trenowanie, ewaluacja i modyfikacja 1|
| 29 Apr - 05 May |==majówka==|
| 06 May - 12 May |trenowanie, ewaluacja i modyfikacja 2|
| 13 May - 19 May |trenowanie, ewaluacja i modyfikacja 3|
| 20 May - 26 May |analiza wyników i przygotowanie artykułu|
| 27 May - 02 Jun |**zgłosznie artykułu (deadline)**|
| 03 Jun - 09 Jun |tydzień na ewentualne dokończenie prac|
| 10 Jun - 16 Jun |==koniec semestru==|


### Planowany zakres eksperymentów
1. Trenowanie architektury Mamba do generowania muzyki w formacie symbolicznym (MIDI, ABC albo MusicXML).
2. Ewaluacja wyników (głównie poprzez ankietyzacje z testem statystycznym).
    - Porównanie muzyki wygenerowanej do prawdziwej oraz wygenerowanej przez inne modele.
    - Przeprowadzenie analizy Mean Opinion Score (MOS).
    - Metryki.
3. (Opcjonalne) Porównanie z transformerem.
4. (Opcjonalnie) Dodanie warunkowania tekstem.
5. Opisanie następnych kroków i możliwych opcji rozwoju.

### Planowany stack technologiczny
1. Python 3
    - autoformatter: black
    - linter: ruff
    - Środowisko wirtualne: Pipenv lub Pip + venv
    - struktura projektu: cookiecutter
3. PyTorch
    - Tensorboard / MLFlow / wandb
    - PyTorch Lightning
5. MidiTok
6. MusPy
7. pretty_midi
8. Git
9. GitHub
10. Huggingface


## Bibliografia
- GU, Albert; DAO, Tri. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.
- GU, Albert, et al. Combining recurrent, convolutional, and continuous-time models with linear state space layers. Advances in neural information processing systems, 2021, 34: 572-585.
- GU, Albert; GOEL, Karan; RÉ, Christopher. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396, 2021.
- SMITH, Jimmy TH; WARRINGTON, Andrew; LINDERMAN, Scott W. Simplified state space layers for sequence modeling. arXiv preprint arXiv:2208.04933, 2022.
- VINAY, Ashvala; LERCH, Alexander. Evaluating generative audio systems and their metrics. arXiv preprint arXiv:2209.00130, 2022.
- YANG, Li-Chia; LERCH, Alexander. On the evaluation of generative models in music. Neural Computing and Applications, 2020, 32.9: 4773-4784.
- FRADET, Nathan, et al. MidiTok: A python package for MIDI file tokenization. arXiv preprint arXiv:2310.17202, 2023.