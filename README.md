# EVA Assignments

# Assignment 4

### The assignment is divided into 3 Experiments

- **Experiment 1** 
    - Strategy 
        - Brute force minimization of number of parameters 
        - Introduction of **GAP** layer
    - Architecture

        <img src="img/gap-layer.svg" width=500 height=500>

    - Logs
        ```
        TRAIN : epoch=0 train_loss=0.03600 correct/total=6330/60000 accuracy=10.6: 100%|██████████| 938/938 [00:22<00:00, 41.71it/s]
        TEST :  epoch=0 test_loss=2.30229 correct/total=980/10000 accuracy=9.8: 100%|██████████| 157/157 [00:03<00:00, 45.71it/s]
        TRAIN : epoch=1 train_loss=0.02464 correct/total=24012/60000 accuracy=40.0: 100%|██████████| 938/938 [00:22<00:00, 41.98it/s]
        TEST :  epoch=1 test_loss=0.41069 correct/total=8779/10000 accuracy=87.8: 100%|██████████|      157/157 [00:03<00:00, 46.81it/s]
        TRAIN : epoch=2 train_loss=0.00340 correct/total=56116/60000 accuracy=93.5: 100%|       ██████████| 938/938 [00:22<00:00, 41.64it/s]
        TEST :  epoch=2 test_loss=0.11479 correct/total=9629/10000 accuracy=96.3: 100%|██████████|      157/157 [00:03<00:00, 46.17it/s]
        TRAIN : epoch=3 train_loss=0.00194 correct/total=57752/60000 accuracy=96.3: 100%|       ██████████| 938/938 [00:22<00:00, 42.17it/s]
        TEST :  epoch=3 test_loss=0.06737 correct/total=9784/10000 accuracy=97.8: 100%|██████████|      157/157 [00:03<00:00, 47.25it/s]
        TRAIN : epoch=4 train_loss=0.00135 correct/total=58414/60000 accuracy=97.4: 100%|       ██████████| 938/938 [00:22<00:00, 42.24it/s]
        TEST :  epoch=4 test_loss=0.05345 correct/total=9849/10000 accuracy=98.5: 100%|██████████|      157/157 [00:03<00:00, 47.20it/s]
        TRAIN : epoch=5 train_loss=0.00108 correct/total=58705/60000 accuracy=97.8: 100%|       ██████████| 938/938 [00:22<00:00, 41.55it/s]
        TEST :  epoch=5 test_loss=0.08425 correct/total=9728/10000 accuracy=97.3: 100%|██████████|      157/157 [00:03<00:00, 46.67it/s]
        TRAIN : epoch=6 train_loss=0.00097 correct/total=58879/60000 accuracy=98.1: 100%|       ██████████| 938/938 [00:22<00:00, 42.20it/s]
        TEST :  epoch=6 test_loss=0.07132 correct/total=9778/10000 accuracy=97.8: 100%|██████████|      157/157 [00:03<00:00, 44.89it/s]
        TRAIN : epoch=7 train_loss=0.00083 correct/total=58996/60000 accuracy=98.3: 100%|       ██████████| 938/938 [00:22<00:00, 41.97it/s]
        TEST :  epoch=7 test_loss=0.03855 correct/total=9877/10000 accuracy=98.8: 100%|██████████|      157/157 [00:03<00:00, 45.47it/s]
        TRAIN : epoch=8 train_loss=0.00073 correct/total=59108/60000 accuracy=98.5: 100%|       ██████████| 938/938 [00:22<00:00, 42.36it/s]
        TEST :  epoch=8 test_loss=0.04076 correct/total=9865/10000 accuracy=98.7: 100%|██████████|      157/157 [00:03<00:00, 47.53it/s]
        TRAIN : epoch=9 train_loss=0.00069 correct/total=59217/60000 accuracy=98.7: 100%|       ██████████| 938/938 [00:21<00:00, 42.66it/s]
        TEST :  epoch=9 test_loss=0.02967 correct/total=9907/10000 accuracy=99.1: 100%|██████████|      157/157 [00:03<00:00, 44.69it/s]
        TRAIN : epoch=10 train_loss=0.00062 correct/total=59261/60000 accuracy=98.8: 100%|      ██████████| 938/938 [00:22<00:00, 42.53it/s]
        TEST :  epoch=10 test_loss=0.03214 correct/total=9888/10000 accuracy=98.9: 100%|██████████|      157/157 [00:03<00:00, 46.52it/s]
        TRAIN : epoch=11 train_loss=0.00059 correct/total=59284/60000 accuracy=98.8: 100%|      ██████████| 938/938 [00:21<00:00, 42.91it/s]
        TEST :  epoch=11 test_loss=0.03589 correct/total=9897/10000 accuracy=99.0: 100%|██████████|      157/157 [00:03<00:00, 47.55it/s]
        TRAIN : epoch=12 train_loss=0.00057 correct/total=59304/60000 accuracy=98.8: 100%|      ██████████| 938/938 [00:22<00:00, 42.50it/s]
        TEST :  epoch=12 test_loss=0.05713 correct/total=9829/10000 accuracy=98.3: 100%|██████████|      157/157 [00:03<00:00, 47.05it/s]
        TRAIN : epoch=13 train_loss=0.00052 correct/total=59368/60000 accuracy=98.9: 100%|      ██████████| 938/938 [00:21<00:00, 43.00it/s]
        TEST :  epoch=13 test_loss=0.02864 correct/total=9911/10000 accuracy=99.1: 100%|██████████|      157/157 [00:03<00:00, 46.41it/s]
        TRAIN : epoch=14 train_loss=0.00050 correct/total=59395/60000 accuracy=99.0: 100%|      ██████████| 938/938 [00:22<00:00, 42.29it/s]
        TEST :  epoch=14 test_loss=0.03484 correct/total=9898/10000 accuracy=99.0: 100%|██████████|      157/157 [00:03<00:00, 46.60it/s]
        TRAIN : epoch=15 train_loss=0.00046 correct/total=59459/60000 accuracy=99.1: 100%|      ██████████| 938/938 [00:21<00:00, 42.85it/s]
        TEST :  epoch=15 test_loss=0.03071 correct/total=9900/10000 accuracy=99.0: 100%|██████████|      157/157 [00:03<00:00, 46.43it/s]
        TRAIN : epoch=16 train_loss=0.00044 correct/total=59472/60000 accuracy=99.1: 100%|      ██████████| 938/938 [00:22<00:00, 42.59it/s]
        TEST :  epoch=16 test_loss=0.03046 correct/total=9910/10000 accuracy=99.1: 100%|██████████|      157/157 [00:03<00:00, 45.61it/s]
        TRAIN : epoch=17 train_loss=0.00044 correct/total=59481/60000 accuracy=99.1: 100%|      ██████████| 938/938 [00:22<00:00, 42.41it/s]
        TEST :  epoch=17 test_loss=0.04340 correct/total=9861/10000 accuracy=98.6: 100%|██████████|      157/157 [00:03<00:00, 46.13it/s]
        TRAIN : epoch=18 train_loss=0.00040 correct/total=59519/60000 accuracy=99.2: 100%|      ██████████| 938/938 [00:22<00:00, 42.02it/s]
        TEST :  epoch=18 test_loss=0.02964 correct/total=9915/10000 accuracy=99.2: 100%|██████████|      157/157 [00:03<00:00, 44.66it/s]
        TRAIN : epoch=19 train_loss=0.00038 correct/total=59520/60000 accuracy=99.2: 100%|      ██████████| 938/938 [00:22<00:00, 42.11it/s]
        TEST :  epoch=19 test_loss=0.03173 correct/total=9903/10000 accuracy=99.0: 100%|██████████|      157/157 [00:03<00:00, 45.59it/s]
        ```

    - Results
      - Feature Visualization
        - Input image
            
            <img src="img/exp1_input.png" width=50 height=50>
        
        - Result
            
            <img src="img/exp1_fv.png" width=500 height=400>

- **Experiment 2** 
    - Strategy 
        - Brute force minimization of number of parameters 
        - Introduction of **GAP** layer
    - Architecture

        <img src="img/gap-layer.svg" width=500 height=500>

    - Logs

    - Results
      - Feature Visualization

- **Experiment 3** 
    - Strategy 
        - Brute force minimization of number of parameters 
        - Introduction of **GAP** layer
    - Architecture

        <img src="img/gap-layer.svg" width=500 height=500>

    - Logs

    - Results
      - Feature Visualization


### Results
