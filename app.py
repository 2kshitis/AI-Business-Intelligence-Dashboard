import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="AI Business Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("AI Business Intelligence")
st.caption("Data Analytics | Data Cleaning | Business Intelligence")

# Upload dataset
uploaded_file = st.file_uploader(
    "Upload CSV or Excel Dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset uploaded successfully!")

    # Dataset overview
    st.header("1. Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Data Cleaning
    st.header("2. Data Cleaning")

    st.subheader("Data Quality Report")

    quality = pd.DataFrame({
        "Missing Values": df.isnull().sum(),
        "Duplicate Values": [
            df[col].duplicated().sum()
            for col in df.columns
        ],
        "Data Type": df.dtypes.astype(str)
    })

    st.dataframe(quality, use_container_width=True)

    st.subheader("Cleaning Options")

    remove_duplicates = st.checkbox(
        "Remove duplicate rows",
        value=True
    )

    clean_column_names = st.checkbox(
        "Clean column names",
        value=True
    )

    fill_missing = st.checkbox(
        "Fill missing values",
        value=True
    )

    if st.button("Clean Dataset", type="primary"):

        clean_df = df.copy()

        # Clean column names
        if clean_column_names:
            clean_df.columns = (
                clean_df.columns
                .astype(str)
                .str.strip()
            )

        # Remove duplicate rows
        if remove_duplicates:
            clean_df = clean_df.drop_duplicates()

        # Fill missing values
        if fill_missing:
            for col in clean_df.columns:

                if clean_df[col].isnull().any():

                    if pd.api.types.is_numeric_dtype(clean_df[col]):
                        median = clean_df[col].median()

                        if pd.notna(median):
                            clean_df[col] = clean_df[col].fillna(median)

                    else:
                        mode = clean_df[col].mode(dropna=True)

                        if not mode.empty:
                            clean_df[col] = clean_df[col].fillna(mode.iloc[0])

        st.session_state["clean_df"] = clean_df

        st.success("Dataset cleaned successfully!")
       
    # 3. BUSINESS ANALYTICS

    st.header("3. Business Analytics")

    # Use cleaned data if available
    analysis_df = st.session_state.get("clean_df", df)

    # Identify numeric columns
    numeric_cols = analysis_df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_cols:
        

        st.subheader("Configure Your Analysis")

        col1, col2 = st.columns(2)

        with col1:
            sales_col = st.selectbox(
                "Select Revenue / Sales Column",
                numeric_cols
            )

            profit_options = ["None"] + numeric_cols

            profit_col = st.selectbox(
                "Select Profit Column",
                profit_options
            )

        with col2:
            product_options = ["None"] + analysis_df.columns.tolist()

            product_col = st.selectbox(
                "Select Product Column",
                product_options
            )

            date_col = st.selectbox(
                "Select Date Column",
                product_options
            )

        # Calculate KPIs
        total_revenue = analysis_df[sales_col].sum()
        average_sales = analysis_df[sales_col].mean()

        if profit_col != "None":
            total_profit = analysis_df[profit_col].sum()
        else:
            total_profit = 0

        # Display KPI cards
        st.subheader("Business Performance")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Revenue",
            f"{total_revenue:,.2f}"
        )

        c2.metric(
            "Total Profit",
            f"{total_profit:,.2f}"
        )

        c3.metric(
            "Average Sales",
            f"{average_sales:,.2f}"
        )

        c4.metric(
            "Total Transactions",
            len(analysis_df)
        )

        # Product analysis
        if product_col != "None":

            st.subheader("Product Performance")

            product_sales = (
                analysis_df.groupby(product_col)[sales_col]
                .sum()
                .sort_values(ascending=False)
            )

            if not product_sales.empty:

                best_product = product_sales.index[0]

                st.success(
                    f"Top Performing Product: {best_product}"
                )

                st.bar_chart(product_sales.head(10))

        # Monthly sales analysis
        if date_col != "None":

            st.subheader("Monthly Sales Analysis")

            monthly_df = analysis_df.copy()

            monthly_df[date_col] = pd.to_datetime(
                monthly_df[date_col],
                errors="coerce"
            )

            monthly_df = monthly_df.dropna(
                subset=[date_col]
            )

            if not monthly_df.empty:

                monthly_df["Month"] = (
                    monthly_df[date_col]
                    .dt.to_period("M")
                    .astype(str)
                )

                monthly_sales = (
                    monthly_df.groupby("Month")[sales_col]
                    .sum()
                    .sort_index()
                )

                st.line_chart(monthly_sales)

        # Profit analysis
        if profit_col != "None":

            st.subheader("Profit Analysis")

            if product_col != "None":

                profit_analysis = (
                    analysis_df.groupby(product_col)[profit_col]
                    .sum()
                    .sort_values(ascending=False)
                )

                st.bar_chart(profit_analysis.head(10))

            
        # ==================================
        # 4. INTERACTIVE VISUALIZATIONS
        # ==================================

        st.header("4. Interactive Visualizations")

        # Interactive product filter
        if product_col != "None":

            products = sorted(
                analysis_df[product_col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_products = st.multiselect(
                "Filter Products",
                products,
                default=products
            )

            visual_df = analysis_df[
                analysis_df[product_col]
                .astype(str)
                .isin(selected_products)
            ].copy()

        else:
            visual_df = analysis_df.copy()

        if visual_df.empty:
            st.warning("No data available for selected filters.")

        else:

            # Revenue by product
            if product_col != "None":

                st.subheader("Revenue by Product")

                revenue_chart = (
                    visual_df.groupby(product_col)[sales_col]
                    .sum()
                    .reset_index()
                    .sort_values(sales_col, ascending=False)
                )

                fig1 = px.bar(
                    revenue_chart,
                    x=product_col,
                    y=sales_col,
                    color=sales_col,
                    title="Product-wise Revenue",
                    hover_data=[sales_col]
                )

                st.plotly_chart(
                    fig1,
                    use_container_width=True
                )

                # Product contribution pie chart
                st.subheader("Product Sales Contribution")

                fig2 = px.pie(
                    revenue_chart,
                    names=product_col,
                    values=sales_col,
                    title="Revenue Distribution",
                    hole=0.4
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

            # Monthly revenue trend
            if date_col != "None":

                st.subheader("Monthly Revenue Trend")

                trend_df = visual_df.copy()

                trend_df[date_col] = pd.to_datetime(
                    trend_df[date_col],
                    errors="coerce"
                )

                trend_df = trend_df.dropna(
                    subset=[date_col]
                )

                if not trend_df.empty:

                    trend_df["Month"] = (
                        trend_df[date_col]
                        .dt.to_period("M")
                        .astype(str)
                    )

                    monthly_data = (
                        trend_df.groupby("Month")[sales_col]
                        .sum()
                        .reset_index()
                        .sort_values("Month")
                    )

                    fig3 = px.line(
                        monthly_data,
                        x="Month",
                        y=sales_col,
                        markers=True,
                        title="Monthly Revenue Growth"
                    )

                    st.plotly_chart(
                        fig3,
                        use_container_width=True
                    )

            # Profit visualization
            if profit_col != "None":

                st.subheader("Profit Distribution")

                fig4 = px.histogram(
                    visual_df,
                    x=profit_col,
                    nbins=20,
                    title="Profit Distribution"
                )

                st.plotly_chart(
                    fig4,
                    use_container_width=True
                )

                # Revenue vs Profit
                fig5 = px.scatter(
                    visual_df,
                    x=sales_col,
                    y=profit_col,
                    color=product_col
                    if product_col != "None"
                    else None,
                    title="Revenue vs Profit",
                    hover_data=visual_df.columns.tolist()
                )

                st.plotly_chart(
                    fig5,
                    use_container_width=True
                )
    
        # ==================================
        # 5. AI BUSINESS INSIGHTS
        # ==================================

        st.header("5. AI Business Insights")

        st.write(
            "Automatically generated business "
            "insights and recommendations."
        )

        insights = []
        recommendations = []

        # Overall sales analysis
        total_sales = analysis_df[sales_col].sum()
        average_sales = analysis_df[sales_col].mean()

        insights.append(
            f"Total recorded revenue is {total_sales:,.2f}."
        )

        insights.append(
            f"Average revenue per record is {average_sales:,.2f}."
        )

        # Product performance
        if product_col != "None":

            product_performance = (
                analysis_df.groupby(product_col)[sales_col]
                .sum()
                .sort_values(ascending=False)
            )

            if not product_performance.empty:

                best_product = product_performance.idxmax()
                worst_product = product_performance.idxmin()

                insights.append(
                    f"{best_product} is the highest revenue "
                    "product."
                )

                insights.append(
                    f"{worst_product} has the lowest total "
                    "revenue among the products."
                )

                recommendations.append(
                    f"Study the sales strategy of {best_product} "
                    "to identify practices that may work for "
                    "other products."
                )

                if len(product_performance) > 1:
                    recommendations.append(
                        f"Investigate the performance of "
                        f"{worst_product} to identify possible "
                        "improvements."
                    )

        # Profit analysis
        if profit_col != "None":

            total_profit = analysis_df[profit_col].sum()

            insights.append(
                f"Total recorded profit is {total_profit:,.2f}."
            )

            if total_profit > 0:
                recommendations.append(
                    "Review costs and pricing to identify "
                    "opportunities to improve profitability."
                )

            elif total_profit < 0:
                recommendations.append(
                    "Investigate operating costs, pricing and "
                    "loss-making transactions."
                )

            else:
                recommendations.append(
                    "Review revenue and cost records to understand "
                    "why recorded profit is zero."
                )

        # Monthly sales analysis
        if date_col != "None":

            date_analysis = analysis_df.copy()

            date_analysis[date_col] = pd.to_datetime(
                date_analysis[date_col],
                errors="coerce"
            )

            date_analysis = date_analysis.dropna(
                subset=[date_col]
            )

            if not date_analysis.empty:

                date_analysis["Month"] = (
                    date_analysis[date_col]
                    .dt.to_period("M")
                )

                monthly_sales = (
                    date_analysis.groupby("Month")[sales_col]
                    .sum()
                    .sort_index()
                )

                if len(monthly_sales) >= 2:

                    first_month = monthly_sales.iloc[0]
                    last_month = monthly_sales.iloc[-1]

                    if first_month != 0:

                        change = (
                            (last_month - first_month)
                            / abs(first_month)
                        ) * 100

                        insights.append(
                            f"Revenue changed by {change:.2f}% "
                            "between the first and last recorded "
                            "months."
                        )

                        if change > 0:
                            recommendations.append(
                                "Analyse the factors behind revenue "
                                "growth and consider applying "
                                "successful strategies."
                            )

                        elif change < 0:
                            recommendations.append(
                                "Investigate the revenue decline "
                                "and review recent sales trends."
                            )

                        else:
                            recommendations.append(
                                "Revenue remained unchanged between "
                                "the first and last recorded months."
                            )

        # Display insights
        st.subheader("Business Findings")

        for i, insight in enumerate(insights, 1):
            st.info(f"Insight {i}: {insight}")

        # Display recommendations
        st.subheader("AI-Generated Recommendations")

        for i, recommendation in enumerate(
            recommendations, 1
        ):
            st.success(
                f"Recommendation {i}: {recommendation}"
            )

        # Export insights
        report_text = (
            "AI BUSINESS INTELLIGENCE REPORT\n\n"
            "BUSINESS FINDINGS\n"
            + "\n".join(insights)
            + "\n\nBUSINESS RECOMMENDATIONS\n"
            + "\n".join(recommendations)
        )

        st.download_button(
            "Download Business Insights Report",
            data=report_text,
            file_name="business_insights.txt",
            mime="text/plain"
        )
        
        # ==================================
        # 6. MACHINE LEARNING PREDICTIONS
        # ==================================

        st.header("6. Machine Learning Sales Prediction")

        st.write(
            "Forecast future monthly sales using "
            "a Linear Regression model."
        )

        if date_col != "None":

            ml_df = analysis_df[[date_col, sales_col]].copy()

            ml_df[date_col] = pd.to_datetime(
                ml_df[date_col],
                errors="coerce"
            )

            ml_df[sales_col] = pd.to_numeric(
                ml_df[sales_col],
                errors="coerce"
            )

            ml_df = ml_df.dropna(
                subset=[date_col, sales_col]
            )

            # Convert daily transactions into monthly sales
            ml_df["Month"] = (
                ml_df[date_col].dt.to_period("M")
            )

            monthly_df = (
                ml_df.groupby("Month")[sales_col]
                .sum()
                .sort_index()
                .reset_index()
            )

            monthly_df["Month"] = (
                monthly_df["Month"].dt.to_timestamp()
            )

            if len(monthly_df) >= 5:

                st.subheader("Forecast Settings")

                forecast_months = st.slider(
                    "How many months do you want to forecast?",
                    min_value=1,
                    max_value=12,
                    value=3
                )

                if st.button(
                    "Train ML Model & Predict",
                    type="primary"
                ):

                    # Create sequential time features
                    X = np.arange(
                        len(monthly_df)
                    ).reshape(-1, 1)

                    y = monthly_df[sales_col].values

                    # Chronological train-test split
                    split_index = int(len(X) * 0.8)
                    split_index = max(
                        2,
                        min(split_index, len(X) - 2)
                    )

                    X_train = X[:split_index]
                    X_test = X[split_index:]

                    y_train = y[:split_index]
                    y_test = y[split_index:]

                    # Train model
                    model = LinearRegression()

                    model.fit(X_train, y_train)

                    # Test predictions
                    test_predictions = model.predict(X_test)

                    mae = mean_absolute_error(
                        y_test,
                        test_predictions
                    )

                    st.session_state["ml_model_results"] = {
                        "model": model,
                        "mae": mae,
                        "r2": (
                            r2_score(y_test, test_predictions)
                            if len(y_test) >= 2
                            else None
                        ),
                        "actual": y_test.tolist(),
                        "predicted": test_predictions.tolist(),
                        "split_index": split_index,
                        "forecast_months": forecast_months,
                        "monthly_df": monthly_df
                    }

                # Display saved model results
                if "ml_model_results" in st.session_state:

                    results = st.session_state[
                        "ml_model_results"
                    ]

                    model = results["model"]

                    if results["forecast_months"] != forecast_months:
                        results["forecast_months"] = forecast_months

                    st.subheader("Model Performance")

                    c1, c2 = st.columns(2)

                    c1.metric(
                        "Mean Absolute Error (MAE)",
                        f"{results['mae']:,.2f}"
                    )

                    if results["r2"] is not None:
                        c2.metric(
                            "R² Score",
                            f"{results['r2']:.3f}"
                        )
                    else:
                        c2.metric("R² Score", "Not available")

                    st.caption(
                        "MAE measures average prediction error. "
                        "R² measures how well the model explains "
                        "variation in the test data."
                    )

                    # Actual vs predicted test results
                    st.subheader("Actual vs Predicted Sales")

                    split_index = results["split_index"]

                    comparison_df = pd.DataFrame({
                        "Month": monthly_df["Month"].iloc[
                            split_index:
                        ],
                        "Actual Sales": results["actual"],
                        "Predicted Sales": results["predicted"]
                    })

                    comparison_long = comparison_df.melt(
                        id_vars="Month",
                        var_name="Type",
                        value_name="Sales"
                    )

                    fig_ml = px.line(
                        comparison_long,
                        x="Month",
                        y="Sales",
                        color="Type",
                        markers=True,
                        title="Actual vs Predicted Monthly Sales"
                    )

                    st.plotly_chart(
                        fig_ml,
                        use_container_width=True
                    )

                    # Future forecasting
                    st.subheader("Future Sales Forecast")

# Prepare training data using all historical months
                    forecast_X = np.arange(
                        len(monthly_df)
                        ).reshape(-1, 1)

                    forecast_y = monthly_df[sales_col].values

# Train final model using all historical data
                    forecast_model = LinearRegression()

                    forecast_model.fit(
                    forecast_X,
                    forecast_y
                    )

# Generate future time periods
                    future_X = np.arange(
                    len(monthly_df),
                    len(monthly_df) + forecast_months
                    ).reshape(-1, 1)

# Predict future sales
                    future_predictions = forecast_model.predict(
                    future_X
                    )

                    future_dates = pd.date_range(
                        start=monthly_df["Month"].iloc[-1]
                        + pd.offsets.MonthBegin(1),
                        periods=forecast_months,
                        freq="MS"
                    )

                    forecast_df = pd.DataFrame({
                        "Month": future_dates,
                        "Predicted Sales": future_predictions
                    })

                    st.dataframe(
                        forecast_df,
                        use_container_width=True
                    )

                    fig_forecast = px.line(
                        forecast_df,
                        x="Month",
                        y="Predicted Sales",
                        markers=True,
                        title="Future Monthly Sales Forecast"
                    )

                    st.plotly_chart(
                        fig_forecast,
                        use_container_width=True
                    )

                    # Download predictions
                    forecast_csv = forecast_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "Download Sales Forecast",
                        data=forecast_csv,
                        file_name="sales_forecast.csv",
                        mime="text/csv"
                    )

            else:
                st.warning(
                    "At least 5 different months of historical "
                    "sales data are required for this model."
                )

        else:
            st.warning(
                "Select a valid date column in Business Analytics "
                "to use Machine Learning predictions."
            )



    else:
        st.warning(
            "No numeric columns found. "
            "Please upload a dataset containing numeric values."
        )


    # Display cleaned dataset
    if "clean_df" in st.session_state:

        clean_df = st.session_state["clean_df"]

        st.subheader("Cleaned Dataset")

        col1, col2, col3 = st.columns(3)

        col1.metric("Original Rows", len(df))
        col2.metric("Cleaned Rows", len(clean_df))
        col3.metric(
            "Remaining Missing Values",
            int(clean_df.isnull().sum().sum())
        )

        st.dataframe(clean_df.head(10), use_container_width=True)

        # Download cleaned dataset
        csv = clean_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Cleaned Dataset",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

else:
    st.info("Upload a dataset to start your analysis.")
