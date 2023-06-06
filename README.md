# PtyNet
Efficient ptychography recontruction strategy by fine tuning of large pre-trained deep learning model

Abstract: With the increasing utilization of pre-trained large models and fine-tuning paradigms in deep learning, the abundance and quality of data have shown significant performance gains for large models. Next-generation light sources, characterized by their high brightness and high coherence nature, offer a vast amount of data that can be leveraged in the field of coherent diffraction imaging, enabling the training of large models in this domain. In this study, we introduce a neural network model and propose enhancements to its architecture. By pre-training the model on extensive datasets and employing a fine-tuning technique, we improve its performance in the reconstruction process. The pre-trained model exhibits remarkable generalization capability, allowing for the reconstruction of diverse sample types, while the fine-tuning technique enhances the quality of the reconstructed results. Additionally, our method demonstrates robust performance across various overlap rates during the reconstruction process. Lastly, we discuss the feasibility of our proposed approach and outline potential avenues for further improvements. We anticipate that our methodology will contribute to the advancement of ptychography and facilitate the development of enhanced imaging techniques in the future.

# Requirements
Pytorch >= 1.6
scipy

# Train, test and finetune
If you want to start training, just use python main.py --model "train". if you want to test , use python main.py --mdoel "test". finetune use python main.py --model "finetune". 

# Simulation of data generation
You can use python create_data.py --model "train" --object_num X --overlap_rate X --creat_data True --concat_data False to generate data with different amounts and overlap rates.

# Others
If you would like to use our data, please contact us via email:panxy@ihep.ac.cn.
