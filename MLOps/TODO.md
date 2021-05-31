1. In evaluate.py, change 'torch.int64' to 'torch.int32'. Becuse version 1.8 of Pytorch is okay but my local version 1.3 is not.
2. Update data and logs parent dir path in utils.py (global variable at the top)
3. In test/test_models.py in generate_data change torch.int64 to torch.int32
4. Properly document functions and classes
5. For dataloading: Use pandas dataframe if doing batch processing i.e. have more than 1 inputs
6. torch.int64 to torch.int32 in inference.py
7. add text preprocessing
8. add inference unti test