from easyeditor import EditTrainer, SERACTrainingHparams, ZsreDataset

#training_hparams = SERACTrainingHparams.from_hparams('/home/u200111312/EasyEdit/hparams/TRAINING/SERAC/gpt-j-6B.yaml')
training_hparams = SERACTrainingHparams.from_hparams('hparams/TRAINING/SERAC/gpt-j-6B.yaml')
train_ds = ZsreDataset('/home/u200111312/EasyEdit/data/data/zsre/zsre_mend_train.json', config=training_hparams)
eval_ds = ZsreDataset('/home/u200111312/EasyEdit/data/data/zsre/zsre_mend_eval.json', config=training_hparams)
trainer = EditTrainer(
    config=training_hparams,
    train_set=train_ds,
    val_set=eval_ds
)
trainer.run()