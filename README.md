---

<div align="center">    
 
# Super Sonic Speed Prototyping Object Detection Research Rapers     


</div>
 
 
## Description   
Object Detection on any data set using supervised learning.
End to end architecture using Pytorch Lightning. 
Compatible with any object detection model as long as they are in pytorch and have a loss/training logic.
Training monitoring using Weights and Bias.


## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project   
cd deep-learning-project-template 
pip install -e .   
pip install -r requirements.txt
 ```   
Get the dataset into folder called data.

Change the paths based on paths on your machine
# Research Playground
python Playground.py    
```
def main():
    #train_df_path on your 
    train_df_path='/data/train.csv'
    seed=42
    seed_everything(seed)
    fold_df,markings=display_(train_df_path,seed=seed)
    bs=64
    fold=2
    model=DETRModel(num_classes=2,num_queries=100)
    c=SetCriterion(1, matcher=HungarianMatcher(), weight_dict={'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}, eos_coef = 0.5, losses=['labels', 'boxes', 'cardinality']).to('cuda')
    run_ = wandb.init(
                        project='e3e3',
                        group=str(fold),
                        name='exp13'
                    )

    Classifier=classifier(
        WheatDataset,
        bs,
        markings,
        fold_df,
        model,
        c,
        run_,
        fold=fold
    )
    
    Trainer=pl.Trainer(devices=1, accelerator="gpu",
                       max_epochs=35,
                      )
    Trainer.fit(Classifier)
```

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
