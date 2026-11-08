# MFCC mean/std audio classification

A self-contained Python / scikit-learn classroom mini-project.

## Install

```bash
pip install numpy scipy librosa scikit-learn matplotlib
```

## Student assignment

```bash
python mfcc_classification_student.py
```

Complete the seven TODO sections. Each audio recording must become a 26-element vector from 13 MFCC means and 13 MFCC standard deviations. Train and evaluate a scikit-learn classifier on stratified 75/25 split.

## Worked solution

```bash
python mfcc_classification_solution.py
```

This writes generated WAV recordings and the confusion-matrix image to `mfcc_project_output/`. A serialized `classifier.pkl` is also saved (only unpickle files you trust).

## Exercises

1. Use only the 13 means, then only the 13 standard deviations. Compare test scores.
2. Change random noise levels or frequency ranges; explain which classes become harder.
3. Replace SVC with LogisticRegression(max_iter=2000) or RandomForestClassifier(random_state=42).
4. Create falling chirps. Discuss whether MFCC means/stds retain time order.
5. Replace generated WAV paths with your labeled real WAV files. Split source groups before windowing when recordings are related.

## Scope

The synthetic sound classes are intentionally easy and a perfect test score on them does not demonstrate real-world audio classification accuracy. Resampling and MFCC calculation use librosa conventions; keep them fixed for inference.
