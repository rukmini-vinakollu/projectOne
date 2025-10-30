import pyodbc
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from plotly.subplots import make_subplots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Streamlit configuration 
st.set_page_config(layout="wide", page_title="Efficient Demand Forecasting for Ride-hailing services")

# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a section', ['Daily Forecast', 'Hourly Forecast', 'Daily EDA', 'Hourly EDA'])

# Define connection string
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=REALME\\SQLEXPRESS;'
    'DATABASE=project;'
    'Trusted_Connection=yes;'
)

if options == "Daily Forecast":

    st.title("Demand Forecast on Daily Basis")
    all_cities = ["Gondor", "Rohan", "Rivendell", "Minas Tirith", "The Shire", "Isengard"]

    # City selection
    selected = st.multiselect(
        "Choose City/Cities to View Forecast",
        options=["Select All"] + all_cities,
        default="Select All"
    )

    # Handle selection logic
    if "Select All" in selected:
        selected_cities = all_cities
    else:
        selected_cities = selected

    # Load data
    for city in selected_cities:
        query = f"""
            SELECT 
                call_date, 
                city, 
                COUNT(*) AS trips 
            FROM project.gold.trips 
            WHERE city = '{city}' 
            GROUP BY call_date, city 
            ORDER BY call_date;
        """
        df = pd.read_sql(query, conn)
        df['call_date'] = pd.to_datetime(df['call_date'])
        df.drop(columns=['city'], inplace=True)
        df.set_index('call_date', inplace=True)
        globals()[f"{city}_daily"] = df

    # Outlier removal
    for city in selected_cities:
        df = globals()[f"{city}_daily"]
        Q1 = df['trips'].quantile(0.25)
        Q3 = df['trips'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df['trips'] = np.where(df['trips'] < lower_bound, lower_bound, df['trips'])
        df['trips'] = np.where(df['trips'] > upper_bound, upper_bound, df['trips'])
        globals()[f"{city}_daily"] = df

    # Holt-Winters parameters
    hw_params = {
        "Gondor": {"smoothing_level": 0.01, "smoothing_trend": 0.1, "smoothing_seasonal": 0.371},
        "Rohan": {"smoothing_level": 0.19, "smoothing_trend": 0.06, "smoothing_seasonal": 0.56},
        "Rivendell": {"smoothing_level": 0.11, "smoothing_trend": 0.01, "smoothing_seasonal": 0.56},
        "Minas Tirith": {"smoothing_level": 0.1, "smoothing_trend": 0.1, "smoothing_seasonal": 0.1},
        "The Shire": {"smoothing_level": 0.053, "smoothing_trend": 0.01, "smoothing_seasonal": 0.05},
        "Isengard": {"smoothing_level": 0.01, "smoothing_trend": 0.1, "smoothing_seasonal": 0.371}
    }

    st.subheader("Forecast Visualization and 2-Day Predictions")

    for city in selected_cities:
        df = globals()[f"{city}_daily"]
        df.index = pd.to_datetime(df.index)
        train_size = int(len(df) * 0.8)
        train, test = df.iloc[:train_size], df.iloc[train_size:]

        train['trips'] = pd.to_numeric(train['trips'], errors='coerce')

        hw_model = ExponentialSmoothing(train['trips'], trend='mul', seasonal='mul', seasonal_periods=7).fit(
            smoothing_level=hw_params[city]["smoothing_level"],
            smoothing_trend=hw_params[city]["smoothing_trend"],
            smoothing_seasonal=hw_params[city]["smoothing_seasonal"]
        )

        total_forecast_len = len(test) + 2
        full_forecast = hw_model.forecast(steps=total_forecast_len)

        test_predictions = full_forecast[:len(test)]
        r2 = r2_score(test['trips'], test_predictions)
        rmse = np.sqrt(mean_squared_error(test['trips'], test_predictions))
        mae = mean_absolute_error(test['trips'], test_predictions)
        mape = np.mean(np.abs((test['trips'] - test_predictions) / test['trips'])) * 100

        st.markdown(f"### {city}")
        st.write(f"**R² Score**: {r2:.4f} | **RMSE**: {rmse:.2f} | **MAE**: {mae:.2f} | **MAPE**: {mape:.2f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train['trips'], mode='lines', name='Training Data', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test.index, y=test['trips'], mode='lines', name='Actual Test Data', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=test.index, y=test_predictions, mode='lines', name='Forecast', line=dict(color='red', dash='dash')))
        fig.update_layout(title=f'Holt-Winters Forecast for {city}',
                          xaxis_title='Date', yaxis_title='Trips',
                          template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Future forecast table
        future_dates = pd.date_range(start=test.index[-1] + pd.Timedelta(days=1), periods=2)
        future_values = full_forecast[-2:].values
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecasted Trips": future_values.astype(int)
        })

        st.write("#### 📅 2-Day Forecast Table")
        st.dataframe(forecast_df)

elif options == "Hourly Forecast":

    st.title("Demand Forecast on Hourly Basis")
    all_cities = ["Gondor", "Rohan", "Rivendell", "Minas Tirith", "The Shire", "Isengard"]
    model_results = {}

    # City selection
    selected = st.multiselect(
        "Choose City/Cities to View Forecast",
        options=["Select All"] + all_cities,
        default="Select All"
    )

    # Handle selection logic
    if "Select All" in selected:
        selected_cities = all_cities
    else:
        selected_cities = selected

    # Load and process data per city
    for city in selected_cities:
        st.subheader(f"📊 Forecast for {city}")

        query = f"SELECT * FROM project.gold.trips_per_hour WHERE city = '{city}' ORDER BY city, call_hour;"
        df_hourly = pd.read_sql(query, conn)
        df_hourly.drop(columns=['city'], inplace=True)
        
        df_hourly['call_hour'] = pd.to_datetime(df_hourly['call_hour'])
        df_hourly['call_date'] = df_hourly['call_hour'].dt.date
        df_hourly.rename(columns={'trips': 'trips_hourly'}, inplace=True)
        
        df_hourly['hour'] = df_hourly['call_hour'].dt.hour
        df_hourly['day_of_week'] = df_hourly['call_hour'].dt.dayofweek
        df_hourly['month'] = df_hourly['call_hour'].dt.month
        df_hourly['lag_168'] = df_hourly['trips_hourly'].shift(168)
        df_hourly['lag_48'] = df_hourly['trips_hourly'].shift(48)
        df_hourly['lag_24'] = df_hourly['trips_hourly'].shift(24)
        df_hourly.dropna(inplace=True)
        df_hourly['hour_day_interaction'] = df_hourly['hour'] * df_hourly['day_of_week']
        df_hourly['day_type'] = df_hourly['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

        daily_totals = df_hourly.groupby('call_date')['trips_hourly'].sum().reset_index()
        daily_totals.rename(columns={'trips_hourly': 'daily_trips'}, inplace=True)
        df_hourly = df_hourly.merge(daily_totals, on='call_date', how='left')
        df_hourly['hourly_proportion'] = df_hourly['trips_hourly'] / df_hourly['daily_trips']

        hourly_patterns = df_hourly.groupby(['day_of_week', 'hour'])['hourly_proportion'].mean().reset_index()
        df_hourly = df_hourly.merge(hourly_patterns, on=['day_of_week', 'hour'], how='left', suffixes=('', '_avg'))

        kmeans = KMeans(n_clusters=5, random_state=42)
        df_hourly['cluster'] = kmeans.fit_predict(df_hourly[['trips_hourly']])
        
        sorted_clusters = np.argsort(kmeans.cluster_centers_.flatten())
        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        cluster_map = dict(zip(sorted_clusters, labels))
        
        df_hourly['demand_type'] = df_hourly['cluster'].map(cluster_map)
        demand_type_encoding = {'Very Low': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
        df_hourly['demand_type_encoded'] = df_hourly['demand_type'].map(demand_type_encoding)

        features = ['hourly_proportion_avg', 'hour_day_interaction', 'lag_24', 'lag_168', 'demand_type_encoded', 'day_type']
        target = 'trips_hourly'

        split_index = int(len(df_hourly) * 0.8)
        X_train = df_hourly.iloc[:split_index][features]
        X_test = df_hourly.iloc[split_index:][features]
        y_train = df_hourly.iloc[:split_index][target]
        y_test = df_hourly.iloc[split_index:][target]
        call_hour_test = df_hourly.iloc[split_index:]['call_hour']

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
        svr_model.fit(X_train_scaled, y_train)
        y_pred = svr_model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred)**0.5
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        model_results[city] = {
            'R² Score': round(r2, 4),
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2),
            'MAPE': round(mape, 2)
        }

        # Plot test prediction
        df_test_plot = pd.DataFrame({
            'Call Hour': call_hour_test,
            'Actual Trips': y_test.values,
            'Predicted Trips': y_pred
        })
        fig = px.line(df_test_plot, x="Call Hour", y=["Actual Trips", "Predicted Trips"],
                    title=f"{city} - Actual vs Predicted (Test Set - SVR)",
                    labels={"value": "Trips", "Call Hour": "Hour"})
        fig.update_traces(hovertemplate='%{x} - %{y}')
        st.plotly_chart(fig)

        # Forecast next 24 hours
        last_known = df_hourly.iloc[-1]
        future_hours = pd.date_range(start=last_known['call_hour'] + pd.Timedelta(hours=1), periods=24, freq='H')
        future_df = pd.DataFrame({'call_hour': future_hours})
        future_df['hour'] = future_df['call_hour'].dt.hour
        future_df['day_of_week'] = future_df['call_hour'].dt.dayofweek
        future_df['month'] = future_df['call_hour'].dt.month
        future_df['hour_day_interaction'] = future_df['hour'] * future_df['day_of_week']
        future_df['day_type'] = future_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        future_df = future_df.merge(hourly_patterns, on=['day_of_week', 'hour'], how='left')
        future_df.rename(columns={'hourly_proportion': 'hourly_proportion_avg'}, inplace=True)
        future_df['lag_168'] = df_hourly['trips_hourly'].iloc[-168:-144].values
        future_df['lag_48'] = df_hourly['trips_hourly'].iloc[-48:-24].values
        future_df['lag_24'] = df_hourly['trips_hourly'].iloc[-24:].values
        future_df['temp_trips'] = df_hourly['trips_hourly'].mean()
        future_df['cluster'] = kmeans.predict(future_df[['temp_trips']].rename(columns={'temp_trips': 'trips_hourly'}))
        future_df['demand_type'] = pd.Series(future_df['cluster']).map(cluster_map)
        future_df['demand_type_encoded'] = future_df['demand_type'].map(demand_type_encoding)

        future_X = future_df[features]
        future_X_scaled = scaler.transform(future_X)
        future_df['predicted_trips'] = svr_model.predict(future_X_scaled)

        fig_forecast = px.line(future_df, x='call_hour', y='predicted_trips',
                            title=f"{city} - Next 24 Hours Forecast (SVR)",
                            labels={"call_hour": "Hour", "predicted_trips": "Predicted Trips"})

        fig_forecast.update_traces(
            hovertemplate='%{x} - %{y:.2f}',
            mode='lines+markers+text',
            text=future_df['predicted_trips'].round(2).astype(str),
            textposition='top center'
        )

        st.plotly_chart(fig_forecast)
        cola, colb = st.columns(2)
        with cola:
            # Show forecast table
            st.markdown(f"**Next 24-Hour Forecast Table for {city}**")
            st.dataframe(future_df[['call_hour', 'predicted_trips', 'demand_type']])
        with colb:
            # Final metrics summary
            st.markdown("### 📈 Model Performance Summary")
            metrics_df = pd.DataFrame.from_dict(model_results, orient='index')
            st.dataframe(metrics_df)

elif options == "Daily EDA":

    st.title("Exploratory Data Analysis on Daily Basis")
    all_cities = ["Gondor", "Rohan", "Rivendell", "Minas Tirith", "The Shire", "Isengard"]

    # City selection
    selected = st.multiselect(
        "Choose City/Cities to View Forecast",
        options=["Select All"] + all_cities,
        default="Select All"
    )

    # Handle selection logic
    if "Select All" in selected:
        selected_cities = all_cities
    else:
        selected_cities = selected

    # Load and visualize data for each selected city
    for city in selected_cities:
        query = f"""
            SELECT 
                call_date, 
                city, 
                COUNT(*) AS trips 
            FROM project.gold.trips 
            WHERE city = '{city}' 
            GROUP BY call_date, city 
            ORDER BY city, call_date;
        """
        df = pd.read_sql(query, conn)

        # Convert 'call_date' to datetime type
        df['call_date'] = pd.to_datetime(df['call_date'])

        # Drop the 'city' column
        df.drop(columns=['city'], inplace=True)

        # Set 'call_date' as index
        df.set_index('call_date', inplace=True)

        # Store in a variable dynamically
        globals()[f"{city}_daily"] = df
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            # Optional: Show basic stats (you can also render this in Streamlit if needed)
            st.subheader(f"Summary Statistics - {city}")
            st.dataframe(df.describe())
        with col2:
            # Create interactive line chart using Plotly
            fig = px.line(
                df, 
                x=df.index, 
                y='trips', 
                markers=True,
                title=f'Trips Over Time - {city}',
                labels={'trips': 'Number of Trips', 'index': 'Date'}
            )

            # Customize layout
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Number of Trips',
                xaxis_tickangle=45,
                template='plotly_white'
            )

            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)

        # Perform decomposition and display interactive plot
        st.subheader(f"Seasonal Decomposition - {city}")

        # Ensure the time series has a regular frequency
        df = df.asfreq('D')  # Assuming daily frequency

        # Fill missing dates with zeros or use interpolation
        df['trips'] = df['trips'].fillna(0)

        # Decompose the time series
        result = seasonal_decompose(df['trips'], model='additive', period=7)

        # Create subplots for observed, trend, seasonal, and residual
        fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))

        fig_decomp.add_trace(go.Scatter(x=df.index, y=result.observed, name='Observed'), row=1, col=1)
        fig_decomp.add_trace(go.Scatter(x=df.index, y=result.trend, name='Trend'), row=2, col=1)
        fig_decomp.add_trace(go.Scatter(x=df.index, y=result.seasonal, name='Seasonal'), row=3, col=1)
        fig_decomp.add_trace(go.Scatter(x=df.index, y=result.resid, name='Residual'), row=4, col=1)

        fig_decomp.update_layout(height=900, title_text=f"Decomposition of Daily Trips - {city}", showlegend=False)

        # Show decomposition in Streamlit
        st.plotly_chart(fig_decomp, use_container_width=True)

        # Add weekday column
        df_box = df.copy()
        df_box = df_box.reset_index()
        df_box['Weekday'] = df_box['call_date'].dt.day_name()

        # Show box plot of trips by weekday
        st.subheader(f"Weekday Trip Distribution - {city}")

        fig_box = px.box(
            df_box,
            x='Weekday',
            y='trips',
            #points='all',  # Show individual data points
            title=f"Distribution of Trips by Weekday - {city}",
            labels={'trips': 'Number of Trips', 'Weekday': 'Day of the Week'},
            color='Weekday'
        )

        # Sort weekdays
        fig_box.update_xaxes(categoryorder='array', categoryarray=[
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ])

        fig_box.update_layout(template='plotly_white')

        # Display box plot in Streamlit
        st.plotly_chart(fig_box, use_container_width=True)


        st.subheader(f"ACF & PACF Plots - Styled for {city}")

        # Set black background style
        plt.style.use("dark_background")

        # Create subplots side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ACF plot
        plot_acf(df['trips'], lags=30, ax=axes[0], zero=False)
        axes[0].set_title("Autocorrelation (ACF)", fontsize=14, color="white")

        # PACF plot
        plot_pacf(df['trips'], lags=30, ax=axes[1], zero=False, method="ywm")
        axes[1].set_title("Partial Autocorrelation (PACF)", fontsize=14, color="white")

        # Improve layout and show in Streamlit
        plt.tight_layout()
        st.pyplot(fig)

else:

    st.title("Exploratory Data Analysis on Hourly Basis")
    all_cities = ["Gondor", "Rohan", "Rivendell", "Minas Tirith", "The Shire", "Isengard"]

    # City selection
    selected = st.multiselect(
        "Choose City/Cities to View Forecast",
        options=["Select All"] + all_cities,
        default="Select All"
    )

    # Handle selection logic
    if "Select All" in selected:
        selected_cities = all_cities
    else:
        selected_cities = selected

    for city in selected_cities:
        query = f"""
        SELECT 
        call_hour,
        city,
        COUNT(*) AS trips
        FROM project.gold.trips 
        WHERE city = '{city}'
        GROUP BY city, call_hour
        ORDER BY city, call_hour;"""
        df_hourly = pd.read_sql(query, conn)
        df_hourly = df_hourly.drop(columns=['city'])
        
        globals()[f"{city.replace(' ', '_')}_hourly"] = df_hourly  # Store as city_hourly

        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            # Optional: Show basic stats (you can also render this in Streamlit if needed)
            st.subheader(f"Summary Statistics - {city}")
            st.dataframe(df_hourly.describe())
        with col2:
            # Create interactive line chart using Plotly
            fig = px.line(
                df_hourly, 
                x=df_hourly.call_hour, 
                y='trips', 
                markers=True,
                title=f'Trips Over Time - {city}',
                labels={'trips': 'Number of Trips', 'call_hour': 'Call_hour'}
            )

            # Customize layout
            fig.update_layout(
                xaxis_title='call_hour',
                yaxis_title='Number of Trips',
                xaxis_tickangle=45,
                template='plotly_white'
            )

            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)

        # Set call_hour as index
        df_hourly.set_index('call_hour', inplace=True)

        # Perform seasonal decomposition (24 for daily pattern in hourly data)
        result = seasonal_decompose(df_hourly['trips'], model='additive', period=24)

        # Create decomposition plot
        fig = go.Figure()

        # Original
        fig.add_trace(go.Scatter(x=df_hourly.index, y=df_hourly['trips'], mode='lines', name='Original'))

        # Trend
        fig.add_trace(go.Scatter(x=df_hourly.index, y=result.trend, mode='lines', name='Trend'))

        # Seasonal
        fig.add_trace(go.Scatter(x=df_hourly.index, y=result.seasonal, mode='lines', name='Seasonality'))

        # Residuals
        fig.add_trace(go.Scatter(x=df_hourly.index, y=result.resid, mode='lines', name='Residuals'))

        # Layout
        fig.update_layout(
            title=f"Time Series Decomposition - {city}",
            xaxis_title="Call Hour",
            yaxis_title="Number of Trips",
            legend_title="Components",
            template="plotly_dark"
        )

        # Display on Streamlit
        #st.subheader(f"Time Series Decomposition - {city}")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("ACF & PACF Plots - Hourly Trips")

        # Use matplotlib dark background style
        plt.style.use("dark_background")

        # Create figure with subplots for ACF and PACF
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ACF plot
        plot_acf(df_hourly['trips'], lags=30, ax=axes[0], zero=False)
        axes[0].set_title("Autocorrelation (ACF)", fontsize=14, color="white")
        axes[0].tick_params(axis='x', colors='white')
        axes[0].tick_params(axis='y', colors='white')

        # PACF plot
        plot_pacf(df_hourly['trips'], lags=30, ax=axes[1], zero=False, method='ywm')
        axes[1].set_title("Partial Autocorrelation (PACF)", fontsize=14, color="white")
        axes[1].tick_params(axis='x', colors='white')
        axes[1].tick_params(axis='y', colors='white')

        # Tweak layout
        plt.tight_layout()

        # Show in Streamlit
        st.pyplot(fig)

        col1, col2 = st.columns([0.3, 0.7])
        #st.subheader("Distribution of Trips by Weekday")
        with col1:

            # Run ADF test on the 'trips' column
            adf_result = adfuller(df_hourly['trips'])

            # Unpack results
            test_statistic = adf_result[0]
            p_value = adf_result[1]
            used_lag = adf_result[2]
            n_obs = adf_result[3]
            crit_values = adf_result[4]

            # Determine conclusion
            if p_value < 0.05:
                conclusion = "Conclusion   : The series is stationary."
            else:
                conclusion = "Conclusion   : The series is non-stationary."

            # Format output
            st.subheader("ADF Test for df_hourly:")
            st.text(f"""
            Test Statistic : {test_statistic:.4f}
            p-value        : {p_value:.4f}
            Used Lags      : {used_lag}
            Number of Observations Used: {n_obs}
            Critical Values:""")
            for key, value in crit_values.items():
                st.text(f"   {key}%: {value:.4f}")
            st.text(conclusion)
       
        with col2:
            # Ensure index is datetime
            df_hourly.index = pd.to_datetime(df_hourly.index)

            # Create a weekday column from the index
            df_hourly['weekday'] = df_hourly.index.day_name()

            # Optional: order weekdays
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            df_hourly['weekday'] = pd.Categorical(df_hourly['weekday'], categories=weekday_order, ordered=True)

            # Create interactive box plot
            fig = px.box(
                df_hourly,
                x='weekday',
                y='trips',
                color='weekday',
                category_orders={'weekday': weekday_order},
                title="Trip Distribution by Day of the Week",
                labels={'trips': 'Number of Trips', 'weekday': 'Weekday'},
                template='plotly_dark'
            )

            fig.update_layout(
                xaxis_title="Day of the Week",
                yaxis_title="Trips",
                showlegend=False
            )

            # Show in Streamlit
            st.plotly_chart(fig, use_container_width=True)

        # Extract hour from datetime index
        df_hourly['hour'] = df_hourly.index.hour

        # Create box plot
        fig = px.box(
            df_hourly,
            x='hour',
            y='trips',
            color='hour',
            title='Trip Distribution Across Hours of the Day',
            labels={'hour': 'Hour of Day (0-23)', 'trips': 'Number of Trips'},
            template='plotly_dark'
        )

        fig.update_layout(
            xaxis=dict(tickmode='linear'),
            xaxis_title='Hour of the Day',
            yaxis_title='Trips',
            showlegend=False
        )

        # Show plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
