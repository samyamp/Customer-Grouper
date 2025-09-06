import streamlit as st
import pandas as pd
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load('machine_model.pkl')
kmeans = model['kmeans_model']
pca = model['pca']
scaler = model['scaler']

