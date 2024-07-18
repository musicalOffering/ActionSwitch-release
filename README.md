"ActionSwitch: Class-agnostic Detection of Simultaneous Actions in Streaming Videos" [ECCV2024] 
=============
[paper]()


Training and assessing the model
=============
1. Check `requirements.txt` and install the necessary packages (there are very few!).

2. Prepare the data (features and labels) in the `data` directory. 
In this code, we will work with the THUMOS14 dataset.
Features and labels are available [here](https://drive.google.com/file/d/1AUyo2YiDYMsU99G18cxWA2BcG-lh59Bf/view?usp=sharing).
Extract the downloaded file into the `data` directory.
After all, the file structure should be as follows:   
    ```
    data
    |--tmp.txt
    |--thumos14
    |  |--thumos14_4state_label_1
    |  |   |--video_test_0000004.npy
    |  |   |--...
    |  |   |--...
    |  |--thumos14_features
    |  |   |--video_test_0000004.npy
    |  |   |--...
    |  |   |--...
    |  |--thumos14_oracle_proposals.pkl
    |  |--thumos14_v2.json
    ```    
3. Download the classifier [here](https://drive.google.com/file/d/1oElmzpwHPxMvyAZtjN-_AWJsm59lCziM/view?usp=sharing), and place it in the `thumos14_classifier_model` directory.

4. [Optional] Download [checkpoint.pt](https://drive.google.com/file/d/1WEdEERZH-uw2yA9YT1fWuXgn30b9X6cX/view?usp=sharing) and place it in the `t_oad_model` directory.
    
5. For training, run the following command:
    ```
    bash train.sh
    ```
    This script will train the model, make proposals with the model, and evaluate the Hungarian F1 score of the model's predictions.
6. To test the given checkpoint, complete step 4 and run the following command:
    ```
    bash test.sh
    ```
    This will result in a `53.2` hungarian f1 score.
