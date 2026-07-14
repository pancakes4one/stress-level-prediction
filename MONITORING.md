# Drift Monitoring Summary

The production data shifted in a way that suggests the model may be seeing a different population than the one it was trained on. In particular, features related to daily behavior and sleep patterns are likely to show the strongest drift because the production batch was generated with changes in distribution rather than being sampled from the same source.

## 1. Which features showed drift and why?

The most likely features to show drift are those that capture lifestyle and routine patterns, such as:

- daily_screen_time_hours
- phone_usage_before_sleep_minutes
- sleep_duration_hours
- sleep_quality_score
- mental_fatigue_score

These features are sensitive to changes in user behavior, habits, or environment. In this project, the production data was created by splitting and altering the original dataset to simulate a new incoming batch, so differences in these variables are expected.

## 2. Would this drift likely affect model performance?

Yes, it could affect performance. If the production distribution differs from the training distribution, the model may see inputs that are less representative of the data it learned from. This is especially relevant for features that strongly influence the prediction of stress level. If the drift is large enough, the model’s accuracy and calibration may degrade.

## 3. Recommended action

I recommend investigating the drift first and then retraining if the drift persists or is confirmed to be meaningful. In practice, this means:

- continue monitoring the drift report for a few more batches
- compare the production feature distributions against the reference data
- retrain the model if the drift remains substantial or if validation performance drops

Overall, the safest next step is to investigate the drift and retrain if the shift is confirmed to be persistent.

