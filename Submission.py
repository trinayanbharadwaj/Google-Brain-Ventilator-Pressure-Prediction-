# Loading the models
model1 = keras.models.load_model("../input/google-brain-ventilator-models/Bi-LSTM-1-val_loss_ 0.2526.h5") 
model2 = keras.models.load_model("../input/google-brain-ventilator-models/Bi-LSTM-val_loss_ 0.2644.h5") 

# predicting
prediction1 = model1.predict(test)
prediction2 = model2.predict(test)

# Converting array to series
prediction1 = prediction1.flatten()
prediction2 = prediction2.flatten()

# Ensembling the output of the 2 models
final_prediction = (prediction1+prediction2)/2

# building a function to create the submission file
def sub(test_prediction):
    df = pd.DataFrame({"id":df_test.id, "pressure":test_prediction})
    return df

submission = sub(final_prediction)

# Saving the submission file
submission.to_csv("submission_ensemble_2_models_final_submission.csv", index=False)
