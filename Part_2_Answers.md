# Part 2: Reasoning-Based Answers

## Q1: Choosing the Right Approach
To identify whether a product is missing its label, I would use object detection.
Detection allows the model to locate the label region and determine whether it exists.
Classification alone would not work well because it does not provide location information.
Segmentation would be more complex than necessary for this task.
If detection does not perform well, my fallback approach would be image classification on cropped regions of interest.

## Q2: Debugging a Poorly Performing Model
First, I would check whether the training and test data come from similar environments.
I would visualize predictions on training vs new factory images to spot domain shift.
Next, I would review label quality and check for annotation errors.
I would also test different confidence thresholds and visualize false positives and false negatives.
Finally, I would try data augmentation or collect more representative data.

## Q3: Accuracy vs Real Risk
Accuracy is not the best metric in this scenario because missing defective products is risky.
I would focus more on recall to ensure defective products are not missed.
Precision is also important, but recall matters more for safety-critical systems.
Metrics like false negative rate and confusion matrix provide better insights.
In such cases, minimizing missed detections is more important than overall accuracy.

## Q4: Annotation Edge Cases
Blurry or partially visible objects should be included if they appear in real-world conditions.
Including them helps the model learn robustness.
However, extremely unclear samples may confuse training and should be reviewed carefully.
The trade-off is between data realism and annotation noise.
A balanced approach improves generalization without degrading performance.
