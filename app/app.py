import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATH CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

# Possible locations for model files
MODEL_PATHS = [
    APP_DIR / "house_price_model.pkl",
    PROJECT_DIR / "house_price_model.pkl",
    PROJECT_DIR / "models" / "house_price_model.pkl",
    PROJECT_DIR / "notebooks" / "house_price_model.pkl",
]

PREPROCESSOR_PATHS = [
    APP_DIR / "house_price_preprocessor.pkl",
    PROJECT_DIR / "house_price_preprocessor.pkl",
    PROJECT_DIR / "models" / "house_price_preprocessor.pkl",
    PROJECT_DIR / "notebooks" / "house_price_preprocessor.pkl",
]


def find_file(paths):
    """Find the first existing file from a list of possible paths."""
    for path in paths:
        if path.exists():
            return path
    return None


# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

model_path = find_file(MODEL_PATHS)
preprocessor_path = find_file(PREPROCESSOR_PATHS)

if model_path is None:
    st.error(
        "❌ house_price_model.pkl was not found.\n\n"
        "Please make sure the trained model file exists."
    )
    st.stop()

if preprocessor_path is None:
    st.error(
        "❌ house_price_preprocessor.pkl was not found.\n\n"
        "Please make sure the preprocessor file exists."
    )
    st.stop()

try:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(59,130,246,0.12), transparent 35%),
            radial-gradient(circle at bottom right, rgba(139,92,246,0.10), transparent 35%),
            #0b1120;
        color: #f8fafc;
    }

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Hero section */
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 24px;
        margin-bottom: 2rem;

        background:
            linear-gradient(
                135deg,
                rgba(30,41,59,0.96),
                rgba(15,23,42,0.96)
            );

        border: 1px solid rgba(148,163,184,0.18);

        box-shadow:
            0 20px 60px rgba(0,0,0,0.30);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 8px;
        color: #ffffff;
    }

    .hero-title span {
        color: #60a5fa;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 17px;
        line-height: 1.6;
    }

    /* Section headers */
    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 1rem;
    }

    .section-description {
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }

    /* Input labels */
    label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }

    /* Input styling */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: rgba(15,23,42,0.85) !important;
        border: 1px solid rgba(148,163,184,0.25) !important;
        border-radius: 10px !important;
    }

    input {
        color: #f8fafc !important;
    }

    /* Prediction card */
    .prediction-card {
        padding: 2.5rem 2rem;
        border-radius: 26px;

        background:
            linear-gradient(
                145deg,
                #172554 0%,
                #1e3a8a 45%,
                #312e81 100%
            );

        border: 1px solid rgba(147,197,253,0.25);

        box-shadow:
            0 25px 70px rgba(30,64,175,0.30);

        text-align: center;

        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .prediction-label {
        color: #bfdbfe;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 12px;
    }

    .prediction-price {
        color: #ffffff;
        font-size: 44px;
        font-weight: 900;
        letter-spacing: -1px;
        margin-bottom: 12px;
    }

    .prediction-note {
        color: #cbd5e1;
        font-size: 14px;
    }

    /* Info cards */
    .info-card {
        padding: 1.2rem;
        border-radius: 16px;

        background: rgba(15,23,42,0.75);

        border: 1px solid rgba(148,163,184,0.15);

        margin-top: 1rem;
    }

    .info-title {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .info-value {
        color: #f8fafc;
        font-size: 20px;
        font-weight: 800;
        margin-top: 4px;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 12px;

        padding: 0.8rem 1rem;

        font-size: 17px;
        font-weight: 700;

        background: linear-gradient(
            135deg,
            #2563eb,
            #7c3aed
        );

        color: white;

        box-shadow:
            0 10px 25px rgba(37,99,235,0.25);

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 15px 35px rgba(37,99,235,0.35);
    }

    /* Divider */
    .divider {
        height: 1px;
        background: rgba(148,163,184,0.15);
        margin: 2rem 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid rgba(148,163,184,0.12);
    }

    .sidebar-title {
        color: #ffffff;
        font-size: 22px;
        font-weight: 800;
    }

    .sidebar-text {
        color: #94a3b8;
        line-height: 1.6;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTION
# AUTOMATIC INDIAN CURRENCY FORMATTING
# ============================================================

def format_indian_price(price):
    """
    Convert model prediction into a user-friendly Indian format.

    Examples:
        12,111,057 -> ₹1.21 Crore
        8,500,000  -> ₹85.00 Lakh
        450,000    -> ₹4.50 Lakh
        75,000     -> ₹75,000
    """

    price = float(price)

    if price >= 1_00_00_000:
        # 1 Crore or more
        crore = price / 1_00_00_000
        return f"₹{crore:,.2f} Crore"

    elif price >= 1_00_000:
        # 1 Lakh or more
        lakh = price / 1_00_000
        return f"₹{lakh:,.2f} Lakh"

    else:
        # Less than 1 Lakh
        return f"₹{price:,.0f}"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">
            🏠 Smart House Predictor
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="sidebar-text">

        This application uses a trained
        <b>Machine Learning model</b> to estimate
        the market value of a property.

        <br><br>

        <b>Model:</b> Linear Regression<br>
        <b>Dataset:</b> 2,500 properties<br>
        <b>Features:</b> 14 property attributes

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="sidebar-text">

        <b>Model Performance</b><br><br>

        R² Score: <b>96.58%</b><br>
        MAPE: <b>4.45%</b><br>
        RMSE: <b>₹627,504.71</b>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
<div class="hero">
    <div class="hero-title">
        🏠 Smart House <span>Price Predictor</span>
    </div>

    <div class="hero-subtitle">
        Estimate the market value of a property using
        machine learning and real estate features.
    </div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# MAIN LAYOUT
# ============================================================

left_column, right_column = st.columns(
    [1.35, 1],
    gap="large"
)


# ============================================================
# LEFT COLUMN — PROPERTY DETAILS
# ============================================================

with left_column:

    st.markdown(
        '<div class="section-title">🏡 Property Details</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Enter the details of the property to generate a price estimate.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # LOCATION / PROPERTY TYPE
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        location = st.selectbox(
            "Location",
            [
                "Downtown",
                "East Beach",
                "West End",
                "Northside",
                "South Suburbs"
            ]
        )

    with col2:
        property_type = st.selectbox(
            "Property Type",
            [
                "Apartment",
                "Villa",
                "House"
            ]
        )

    # --------------------------------------------------------
    # AREA / BEDROOMS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        area_sqft = st.number_input(
            "Area (Sq Ft)",
            min_value=100,
            max_value=20000,
            value=1371,
            step=50
        )

    with col2:
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )

    # --------------------------------------------------------
    # BATHROOMS / FLOORS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5
        )

    with col2:
        floors = st.number_input(
            "Floors",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )

    # --------------------------------------------------------
    # PROPERTY AGE / PARKING
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        property_age = st.number_input(
            "Property Age (Years)",
            min_value=0,
            max_value=100,
            value=5,
            step=1
        )

    with col2:
        parking_spaces = st.number_input(
            "Parking Spaces",
            min_value=0,
            max_value=10,
            value=1,
            step=1
        )

    # --------------------------------------------------------
    # DISTANCE / SCHOOLS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        distance_to_center = st.number_input(
            "Distance to City Center (Km)",
            min_value=0.0,
            max_value=100.0,
            value=7.2,
            step=0.1
        )

    with col2:
        nearby_schools = st.number_input(
            "Nearby Schools",
            min_value=0,
            max_value=20,
            value=4,
            step=1
        )

    # --------------------------------------------------------
    # CONDITION / LOCATION RATING
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        condition_score = st.slider(
            "Condition Score",
            min_value=1,
            max_value=10,
            value=6
        )

    with col2:
        location_rating = st.slider(
            "Location Rating",
            min_value=1,
            max_value=10,
            value=8
        )

    # --------------------------------------------------------
    # FURNISHING / PRICE PER SQFT
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        furnishing_status = st.selectbox(
            "Furnishing Status",
            [
                "Unfurnished",
                "Semi-Furnished",
                "Furnished"
            ]
        )

    with col2:
        price_per_sqft = st.number_input(
            "Price per Sq Ft",
            min_value=100,
            max_value=100000,
            value=8902,
            step=100
        )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_button = st.button(
        "🔮 Predict House Price"
    )


# ============================================================
# RIGHT COLUMN — PREDICTION
# ============================================================

with right_column:

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Your estimated property value will appear here.'
        '</div>',
        unsafe_allow_html=True
    )

    if predict_button:

        try:

            # ------------------------------------------------
            # CREATE INPUT DATAFRAME
            # ------------------------------------------------

            input_data = pd.DataFrame(
                {
                    "Location": [location],
                    "Property_Type": [property_type],
                    "Area_SqFt": [area_sqft],
                    "Bedrooms": [bedrooms],
                    "Bathrooms": [bathrooms],
                    "Floors": [floors],
                    "Property_Age_Years": [property_age],
                    "Parking_Spaces": [parking_spaces],
                    "Distance_to_City_Center_Km": [distance_to_center],
                    "Nearby_Schools": [nearby_schools],
                    "Condition_Score": [condition_score],
                    "Location_Rating": [location_rating],
                    "Furnishing_Status": [furnishing_status],
                    "Price_per_SqFt": [price_per_sqft]
                }
            )

            # ------------------------------------------------
            # TRANSFORM INPUT
            # ------------------------------------------------

            transformed_data = preprocessor.transform(
                input_data
            )

            # ------------------------------------------------
            # MAKE PREDICTION
            # ------------------------------------------------

            prediction = model.predict(
                transformed_data
            )[0]

            # ------------------------------------------------
            # FORMAT PRICE AUTOMATICALLY
            # ------------------------------------------------

            formatted_price = format_indian_price(
                prediction
            )

            # ------------------------------------------------
            # PREDICTION CARD
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="prediction-card">

                    <div class="prediction-label">
                        ESTIMATED MARKET VALUE
                    </div>

                    <div class="prediction-price">
                        {formatted_price}
                    </div>

                    <div class="prediction-note">
                        Estimated using the trained machine learning model
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # ADDITIONAL INFORMATION
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">'
                '📌 Property Summary'
                '</div>',
                unsafe_allow_html=True
            )

            info1, info2 = st.columns(2)

            with info1:

                st.markdown(
                    f"""
                    <div class="info-card">

                        <div class="info-title">
                            Property Area
                        </div>

                        <div class="info-value">
                            {area_sqft:,} Sq Ft
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with info2:

                st.markdown(
                    f"""
                    <div class="info-card">

                        <div class="info-title">
                            Bedrooms
                        </div>

                        <div class="info-value">
                            {bedrooms}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            info3, info4 = st.columns(2)

            with info3:

                st.markdown(
                    f"""
                    <div class="info-card">

                        <div class="info-title">
                            Location
                        </div>

                        <div class="info-value">
                            {location}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with info4:

                st.markdown(
                    f"""
                    <div class="info-card">

                        <div class="info-title">
                            Property Type
                        </div>

                        <div class="info-value">
                            {property_type}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ------------------------------------------------
            # RAW MODEL PREDICTION
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="info-card">

                    <div class="info-title">
                        Model Prediction
                    </div>

                    <div class="info-value">
                        ₹{prediction:,.0f}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:

            st.error(
                f"❌ Prediction failed: {e}"
            )

    else:

        st.markdown(
            """
<div class="prediction-card">
    <div class="prediction-label">
        READY TO PREDICT
    </div>

    <div class="prediction-price">
        🏠
    </div>

    <div class="prediction-note">
        Enter the property details and click
        "Predict House Price"
    </div>
</div>
""",
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "<div class='divider'></div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        font-size:13px;
        padding-bottom:1rem;
    ">
        Smart House Price Predictor • Machine Learning Project
        <br>
        Built with Python, Scikit-learn & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)