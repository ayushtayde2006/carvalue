# 🚗 CarValue: Used Car Price Prediction

An ML-powered resale price estimator. Enter a used car's details and get an estimated price with a likely range, using **multiple linear regression** on a log-transformed target.

> **Important:** the output is an *estimate*, not a guaranteed selling price. The model learned from historical **listing prices (about 2020)**, which differ from real transaction prices and from today's market.

![alt text](image-2.png)

---

## Problem statement

Used-car buyers and sellers struggle to judge a fair resale price. Price depends on age, mileage, brand and model, fuel type, transmission, ownership history, and more. This project learns those relationships from historical listings and estimates the price of a car the model has not seen.

## Objectives

- Predict a reasonable resale price from multiple car features
- Identify the factors associated with higher or lower prices
- Evaluate prediction errors on unseen data
- Provide a simple interface for entering car details
- Explain the limitations and reliability of the estimates

## Dataset

- Kaggle used-car listings dataset (about 4,300 rows, cars from 1992 to 2020)
- Columns: `name, year, selling_price, km_driven, fuel, seller_type, transmission, owner`
- Source: `<https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?resource=download>`
- The raw file is not included in this repo. Download it from the link above.

**Data audit.** An earlier, larger dataset (about 140,000 rows) was rejected because it looked synthetic: median price was flat across year bands, identical models and years had up to a 19x price spread, and the R² stayed near 0.19. The dataset used here passed the same checks: price rises with year (correlation 0.41) and falls with kilometres driven (-0.19), and old cars are cheap.

## Approach

1. **Cleaning:** removed duplicate rows, standardised text, dropped invalid values, and capped extreme `km_driven` at the 99.5th percentile. Prices were kept because luxury cars are real; the log transform handles the skew.
2. **Feature engineering:**
   - `car_age = 2026 - year`
   - `brand` (first word of the name) and `model` (first two words); categories with fewer than 15 rows grouped as `other`
   - `owner_num`: ordered integer encoding of ownership
   - One-hot encoding for brand, model, fuel, transmission, seller type (first category dropped to avoid the dummy variable trap)
3. **Honest evaluation:** train/test split before any scaling or encoding (80/20, fixed seed), a mean-price baseline, and shuffled 5-fold cross-validation.
4. **Target transformation:** price is right-skewed, so the model predicts `log(price)` and converts back with `exp`.
5. **Pipeline:** a scikit-learn `Pipeline` with `ColumnTransformer` (scaling + one-hot encoding + `LinearRegression`), so preprocessing and the model are saved together.

## Results

All figures are on the held-out test set unless stated otherwise. Prices are in Indian rupees (₹).

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (always predict mean price) | 298,075 | 508,258 | -0.001 |
| Linear regression, raw price | 177,051 | 301,416 | 0.648 |
| Linear regression, log price (brand only) | 133,377 | 250,009 | 0.758 |
| **Linear regression, log price + car model (final)** | **91,324** | **196,050** | **0.851** |
| + log-transformed km (no gain, dropped) | 92,054 | 196,704 | 0.850 |
| Ridge α=0.1 (no gain) | 91,464 | 196,499 | 0.850 |
| Ridge α=1 | 96,799 | 214,188 | 0.822 |

**Cross-validated figures (more reliable than a single split):**

- Shuffled 5-fold CV on the training set: mean R² **0.766**, mean MAE about **₹101,800**
- Cross-validated predictions over all data: MAE **₹91,785**, RMSE ₹219,475, R² **0.815**
- About **80% of true prices fall between 0.72x and 1.36x the estimate**. The app shows this as the "likely range".

**Why Linear Regression, not Ridge:** Ridge never beat plain linear regression, and got worse as regularisation increased, so multicollinearity was not hurting the model.

**Error by price band (test set):**

| Price band | Mean absolute error | Median % error |
|---|---|---|
| Low | ₹35,158 | 21% |
| Mid-low | ₹57,090 | 16% |
| Mid-high | ₹74,221 | 13% |
| High | ₹206,022 | 14% |

Errors grow with price in rupees, but stay about 13-21% in percentage terms. Cheap cars have the largest relative error, since condition matters most at the low end and the dataset has no condition information.

**Assumption checks (log model):** residuals form a random cloud around zero with no strong curve, and their distribution is roughly bell-shaped, so linearity, constant variance and normality hold reasonably well.

## What drives price

Effects from the log-price linear regression, holding the other features constant:

| Rank | Factor | Effect on price |
|---|---|---|
| 1 | **Car model / brand** | Largest effect: about -94% (Tata Nano) to +102% (Land Rover), relative to an Audi A4 |
| 2 | **Age** | about **-9.7%** per extra year |
| 3 | **Transmission** | Manual about **16% cheaper** than automatic |
| 4 | **Fuel** | Diesel about **15% higher** than CNG; petrol and LPG about the same |
| 5 | **Ownership** | about **-4%** per previous owner |
| 6 | **Seller type** | Individual sellers about **8.5% lower** than dealers |
| 7 | **Kilometres driven** | about **-1.5%** per 10,000 km (small, because age already captures most wear) |

Sanity check: the cheapest models are Tata Nano, Ford Ikon, Datsun Redigo, Chevrolet Spark and Maruti 800; the most expensive are Land Rover, Audi Q5, BMW 5-Series, Mercedes E-Class and Toyota Fortuner. Both lists match the real used-car market.

Notes on reading the coefficients:

- **Brand and model overlap** (each model belongs to one brand), so a single coefficient can't be read alone. The net model effect is the brand coefficient plus the model coefficient. Predictions are unaffected.
- Model comparisons are relative to the dropped reference category (Audi A4).
- Electric cars have very few rows, so their effect (-37%) is unreliable.
- These are associations, not causes. Age and kilometres are correlated, which is why kilometres looks weak.

## Run the app

```bash
git clone https://github.com/ayushtayde2006/carvalue.git
cd carvalue
pip install -r requirements.txt
streamlit run app.py
```

The repo includes the trained pipeline (`carvalue_model.joblib`) and app metadata (`meta.joblib`), so you don't need to retrain. Use the library versions in `requirements.txt`, because a model saved with one scikit-learn version may not load in another.

**App features**

- Dropdowns for car model, fuel, transmission, seller type and ownership; inputs for year and kilometres
- Brand filled in automatically from the chosen model, so category names always match training
- Estimated price plus a likely range
- Warnings for years after 2020 or before 2000, and for kilometres beyond the training range
- Visible disclaimer

**Sample outputs** (used as sanity checks): Maruti 800 about ₹83k, Maruti Swift about ₹6 lakh, Toyota Fortuner about ₹26 lakh, BMW 5-Series about ₹25 lakh.

## Project structure

```
carvalue/
├── app.py                    # Streamlit app
├── carvalue_model.joblib     # trained pipeline (preprocessing + model)
├── meta.joblib               # dropdown values, ranges, error bounds
├── requirements.txt
├── notebooks/                # training notebook (optional)
├── screenshot.png
└── README.md
```

## Limitations

- **Listing prices, not transaction prices.** Actual sale prices are usually lower after negotiation.
- **Old data.** The data is from about 2020, so estimates are lower than today's market.
- **No cars under 6 years old.** The newest training car is from 2020, so newer cars are outside the training range and estimates for them are unreliable.
- **Missing information.** The model can't see condition, accident history, colour, variant, engine size, location or local demand.
- **Rare and luxury models** are grouped as `other` or have few examples, and show the largest errors (up to 60-70% on the worst cases).
- **Wide uncertainty.** Typical error is about ₹92k (13-21% of price), and about 80% of true prices fall between 0.72x and 1.36x the estimate.
- **Model assumptions.** Linear regression assumes straight-line effects on log price, which real pricing only approximates.
- **Noisy coefficients for small categories** (for example electric cars), and overlapping brand and model effects.
- **Possible data-entry errors** (for example very low kilometres on old cars) remain in the data.
- **Prices drift over time**, so the model needs retraining on fresh data.
- **Minor leakage:** rare-category grouping used the full dataset before splitting. The effect is small.

## Possible improvements

- Use a larger, more recent dataset with engine size, power and variant
- Clean model names (for example the "mercedes-benz new" artifact from taking the first two words)
- Try non-linear models such as Random Forest to measure what linear regression leaves out
- Retrain periodically on fresh listings

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib, Streamlit. Developed in Google Colab.

