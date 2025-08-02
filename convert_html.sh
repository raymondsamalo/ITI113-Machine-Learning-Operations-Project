#!/usr/bin/env bash
# This script is used to convert Jupyter Notebook to various formats using nbconvert.
# jupyter nbconvert --to script AndroidMalwareRaymondSamalo.ipynb
# jupyter nbconvert --to webpdf --allow-chromium-download $1
jupyter nbconvert --to html  $1