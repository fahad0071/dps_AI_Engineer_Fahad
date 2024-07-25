# DPS AI Engineer TASK (FAHAD ZAHID)


## Step 1:
- Explored data and Preprocced the data using Pandas

## Step 2: 
- Created Visualizations using Matplotlib and seaborn

## Step 3:
- Created Machine Learning Model(Random Forrest Regressor)
- R2 score and MSE for evaluation


## For deployment:


### Created a flask app:
- Load pre-trained Random Forest model (random_forest_model.pkl) and two label encoders (label_encoder_monatszahl.pkl and label_encoder_auspraegung.pkl)

## Created Prediction Endpoint

- The /predict endpoint handles POST requests.
- It expects JSON input with the keys Category, Type, Year, and Month.
- The input data is transformed into a DataFrame, with the categorical features being encoded using the loaded label encoders.
- The model makes predictions based on the transformed data.
- The response is formatted to return a single prediction or an error message if multiple predictions are received.

## Error Handling:
- The endpoint has a try-except block to handle exceptions, returning appropriate error messages and example input format.

## Created Docker File
- Build and Test the Docker Image Locally

## Deployment on Azure:
- Login to Azure
- Create a Resource Group
- Create an Azure Container Registry
- Login to the ACR
- Push the Docker image to ACR
- Create an Azure Container Apps environment
- Deploy the container to Azure Container Apps
- Retrieve the public URL of your deployed app
- Verify the Deployment

## Validate using Postman

## Example

#### Sample Input:
{
    "Category": "Fluchtunfälle",
    "Type": "insgesamt",
    "Year": "2006",
    "Month": "06"
}

#### Sample Return:

{
    "Category": "Fluchtunfälle",
    "Type": "insgesamt",
    "Year": "2006",
    "Month": "06"
}


## I thoroughly enjoyed working on this task; it was a great way to brush up on concepts.

