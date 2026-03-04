# Model Card Template

## Model Name
- 

## Task
- Multiclass corporate default prediction (`0=survive, 1=default, 2=other exit`)
- Horizon(s): 

## Data
- Source: AIDF-CRI oil industry panel data
- Training window: 1991-2014
- Test window: 2015-2025
- Frequency: Monthly
- Features used:
- Preprocessing:

## Model
- Algorithm:
- Class imbalance handling:
- Hyperparameter tuning approach:
- Reproducibility seed:

## Evaluation
- Primary metric: default-vs-rest AUC (`y==1`)
- Secondary metrics:
- Overall AUC:
- Mean yearly AUC:
- Valid years:

## Runtime & Submission
- Estimated train + predict time:
- Dependencies:
- Output file format: `predictions.csv` with `CompNo,yyyy,mm,p0,p1,p2`

## Limitations
- 

## Intended Use
- 

## Author
- 
