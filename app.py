# ------------------------- Product Reviews Dashboard -------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# ------------------------- Page Config & CSS -------------------------
st.set_page_config(page_title="Product Reviews Dashboard", layout="wide", page_icon=":bar_chart:")

st.markdown("""
    <style>
        div.block-container { padding-top: 1rem; }
        .stat-card {
            border: 2px solid rgba(160, 160, 160, 0.3);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            background-color: rgba(200, 200, 200, 0.2);
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        }
        .stat-value {
            font-size: 36px;
            font-weight: bold;
            color: #674FEE;
        }
        .custom-hr {
            border: none;
            height: 2px;
            background: linear-gradient(to right, #F7418F, #FFCBCB, #3AA6B9);
            margin: 40px 0;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="display: flex; align-items: center; justify-content: center;">
        <h1 style="font-family: 'Courier New', Courier, monospace; font-weight: bold; font-size: 60px;">Product Reviews Dashboard</h1>
    </div>
""", unsafe_allow_html=True)

# ------------------------- Load Data from Dropbox -------------------------
url = "https://drive.google.com/uc?export=download&id=1L17YWwxNha2OqNMDl21urHU9ckHmGEvY"
df = pd.read_csv(url)

# Convert Time to datetime & extract Year, Month, Day
df['Time'] = pd.to_datetime(df['Time'])
df['Year'] = df['Time'].dt.year
df['Month'] = df['Time'].dt.month
df['Day'] = df['Time'].dt.day

# ------------------------- Sidebar -------------------------
st.sidebar.image("shopping.png", width=200)
st.sidebar.header("Choose your filters:")

# Year range selectboxes
years = sorted(df['Year'].unique())
start_years = years[:-1]
end_years = years

col1, col2 = st.sidebar.columns(2)
with col1:
    start_year = st.selectbox("Start Year", start_years, index=0)
with col2:
    end_year = st.selectbox("End Year", end_years, index=len(end_years)-1)

if start_year > end_year:
    st.sidebar.warning("⚠️ Start year must be less than or equal to end year.")
    st.stop()

filtered_df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

# Score filter
scores = sorted(df['Score'].unique())
selected_scores = st.sidebar.multiselect("Select Score(s)", scores)
if selected_scores:
    filtered_df = filtered_df[filtered_df['Score'].isin(selected_scores)]

# ------------------------- KPI Cards -------------------------
total_products = filtered_df['ProductId'].nunique()
total_users = filtered_df['UserId'].nunique()
total_reviews = filtered_df.shape[0]

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<h3 style='text-align:center;font-size:20px;font-family:\"Courier New\";'>Total Products</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='stat-card'><div class='stat-value'>{total_products}</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<h3 style='text-align:center;font-size:20px;font-family:\"Courier New\";'>Total Users</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='stat-card'><div class='stat-value'>{total_users}</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown("<h3 style='text-align:center;font-size:20px;font-family:\"Courier New\";'>Total Reviews</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='stat-card'><div class='stat-value'>{total_reviews}</div></div>", unsafe_allow_html=True)

st.markdown('<hr class="custom-hr">', unsafe_allow_html=True)

# ------------------------- Sentiment Trend & Donuts for Diverse Users -------------------------
col1, col2 = st.columns([2, 1])  

# ---------- Line Chart: Sentiment Trend for Diverse Users ----------
with col1:
    diverse_filtered = filtered_df[filtered_df['Behavior'] == 'Diverse']

    if diverse_filtered.empty:
        st.info("No Diverse users found for the selected filters.")
    else:
        yearly_sentiment = (
            diverse_filtered
            .groupby('Year')['Sentiment']
            .value_counts(normalize=True)
            .unstack()
            .fillna(0)
            .reset_index()
        )

        yearly_sentiment['Year'] = yearly_sentiment['Year'].astype(int)

        y_columns = [
            col for col in ['positive','negative','neutral']
            if col in yearly_sentiment.columns and yearly_sentiment[col].sum() > 0
        ]

        st.markdown(
            "<h3 style='text-align:center;font-size:20px;font-family:\"Courier New\";'>Sentiment Trend of Diverse Users Over Years</h3>",
            unsafe_allow_html=True
        )

        sentiment_colors = {'positive': '#674FEE','negative': 'red','neutral': 'gray'}

        fig_diverse = px.line(
            yearly_sentiment,
            x='Year',
            y=y_columns,
            markers=True,
            labels={'value':'Fraction of Reviews','variable':'Sentiment'},
            color_discrete_map=sentiment_colors
        )
        fig_diverse.update_traces(line=dict(width=2), marker=dict(size=6))
        fig_diverse.update_yaxes(range=[0, 1])
        fig_diverse.update_xaxes(dtick=1)  
        fig_diverse.update_layout(height=350, margin=dict(l=10,r=10,t=10,b=10))

        st.plotly_chart(fig_diverse, use_container_width=True)

# ---------- Donut Charts: Sentiment & Behavior ----------
with col2:
    st.markdown(
        "<h3 style='text-align: center; font-size: 20px; font-family: \"Courier New\", Times, serif;'>Customer Sentiment</h3>",
        unsafe_allow_html=True
    )
    sentiment_counts = filtered_df['Sentiment'].value_counts()
    sentiment_colors = {'positive': '#674FEE','negative': "#C50101",'neutral': "#A5A8A8"}
    fig_sentiment = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        hole=0.6,
        color=sentiment_counts.index,
        color_discrete_map=sentiment_colors
    )
    fig_sentiment.update_traces(textinfo='percent', textfont_size=12)
    fig_sentiment.update_layout(height=130, margin=dict(l=5,r=5,t=6,b=5))
    st.plotly_chart(fig_sentiment, use_container_width=True, key="sentiment_donut")

    st.markdown(
        "<h3 style='text-align: center; font-size: 20px; font-family: \"Courier New\", Times, serif;'>Customer Behavior</h3>",
        unsafe_allow_html=True
    )
    behavior_counts = filtered_df['Behavior'].value_counts()
    behavior_colors_list = ['#674FEE',  "#3223FA", "#C50101", "#A5A8A8"]
    behavior_colors = {behavior_counts.index[i]: behavior_colors_list[i] for i in range(len(behavior_counts))}
    
    fig_behavior = px.pie(
        values=behavior_counts.values,
        names=behavior_counts.index,
        hole=0.6,
        color=behavior_counts.index,
        color_discrete_map=behavior_colors
    )
    fig_behavior.update_traces(textinfo='percent', textfont_size=12)
    fig_behavior.update_layout(height=130, margin=dict(l=5,r=5,t=6,b=5))
    st.plotly_chart(fig_behavior, use_container_width=True, key="behavior_donut")

st.markdown('<hr class="custom-hr">', unsafe_allow_html=True)

# ------------------------- Bar Charts: Top & Bottom Products -------------------------
def format_title(title, max_words_per_line=5):
    words = str(title).split()
    lines = [' '.join(words[i:i+max_words_per_line]) for i in range(0,len(words),max_words_per_line)]
    return '<br>'.join(lines)

# ------- Calculate number of reviews and average score per product -------
product_stats = filtered_df.groupby('ProductId').agg(
    num_reviews=('Score','count'),
    avg_score=('Score','mean')
).reset_index()

# ------- Normalize to prepare for Joint Metric -------
product_stats['norm_reviews'] = (product_stats['num_reviews'] - product_stats['num_reviews'].min()) / (product_stats['num_reviews'].max() - product_stats['num_reviews'].min())
product_stats['norm_avg_score'] = (product_stats['avg_score'] - product_stats['avg_score'].min()) / (product_stats['avg_score'].max() - product_stats['avg_score'].min())

# ------- Adjust joint metric to handle identical avg_score -------
if product_stats['avg_score'].nunique() == 1:
    # All avg_scores are the same, weight more on number of reviews
    product_stats['joint_metric'] = product_stats['norm_reviews'] * 2  
else:
    # Normal joint metric
    product_stats['joint_metric'] = product_stats['norm_reviews'] + product_stats['norm_avg_score']

# ------- Top & Bottom Products -------
top_products = product_stats.sort_values('joint_metric', ascending=False).head(10)
bottom_products = product_stats.sort_values('joint_metric', ascending=True).head(10)

# ------- Format Titles -------
top_products['formatted'] = top_products['ProductId'].apply(format_title)
bottom_products['formatted'] = bottom_products['ProductId'].apply(format_title)

# ------- Plot Bar Charts -------
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 style='text-align:center;font-size:20px;font-family:\"Courier New\";'>Top 10 Products by Reviews & Avg Score</h3>", unsafe_allow_html=True)

    purple_scale = [
        [0.0, "#E5D9FF"],
        [0.3, "#B39CFF"],
        [0.6, "#8A6EFF"],
        [0.8, "#674FEE"],
        [1.0, "#4B34D7"]
    ]

    fig_top = px.bar(
        top_products.sort_values('joint_metric'),
        x='num_reviews',
        y='formatted',
        orientation='h',
        color='avg_score',
        color_continuous_scale=purple_scale,
        labels={
            'num_reviews': 'Number Of Reviews',
            'formatted': 'ProductId',
            'avg_score': 'Avg Score'
        }
    )

    fig_top.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_showscale=True
    )

    st.plotly_chart(fig_top, use_container_width=True)

with col2:
    st.markdown("<h3 style='text-align:center;font-size:20px;font-family:\"Courier New\";'>Worst 10 products by Reviews & Avg Score</h3>", unsafe_allow_html=True)

    red_scale = [
        [0.0,  "#FFD6D6"],  
        [0.25, "#FF8A8A"],  
        [0.5,  "#E04444"],  
        [0.75, "#C50101"],  
        [1.0, "#6A0000"]    
    ]

    fig_bottom = px.bar(
        bottom_products.sort_values('joint_metric'),
        x='num_reviews',
        y='formatted',
        orientation='h',
        color='avg_score',
        color_continuous_scale=red_scale,
        labels={
            'num_reviews': 'Number Of Reviews',
            'formatted': 'ProductId',
            'avg_score': 'Avg Score'
        }
    )

    fig_bottom.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_showscale=True
    )

    st.plotly_chart(fig_bottom, use_container_width=True)

st.markdown('<hr class="custom-hr">', unsafe_allow_html=True)

# ------------------------- Line Chart & Scatter Side by Side -------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 style='text-align:center;font-size:20px;font-family:\"Courier New\";'>Average Score by Year</h3>", unsafe_allow_html=True)
    avg_score_year = filtered_df.groupby('Year')['Score'].mean().reset_index()
    fig_line = px.line(avg_score_year, x='Year', y='Score', markers=True)
    fig_line.update_traces(line=dict(color="#674FEE", width=2), marker=dict(size=6))
    fig_line.update_layout(height=350, margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig_line, use_container_width=True)

with col2:
    product_avg = filtered_df.groupby('ProductId')['Score'].mean().reset_index(name='avg_score')
    product_counts = filtered_df.groupby('ProductId')['Score'].count().reset_index(name='num_reviews')
    product_stats = pd.merge(product_avg, product_counts, on='ProductId')

    st.markdown("<h3 style='text-align:center;font-size:20px;font-family:\"Courier New\";'>Number of Reviews vs Avg Score Per Products</h3>", unsafe_allow_html=True)
    fig_scatter = px.scatter(
        product_stats,
        x='num_reviews',
        y='avg_score',
        hover_name='ProductId',
        color_discrete_sequence=["#674FEE"]
    )
    fig_scatter.update_traces(marker=dict(size=6, opacity=0.6))
    fig_scatter.update_layout(height=350, margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig_scatter, use_container_width=True)





