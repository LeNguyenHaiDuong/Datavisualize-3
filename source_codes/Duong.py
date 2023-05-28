import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title = "Doanh thu hang thang cua cac rap chieu phim", layout = 'wide', 
                   initial_sidebar_state = 'expanded')

data = pd.read_csv("../datasets/data_preprocess.csv", index_col = 0)

### Change type
data['date'] = pd.to_datetime(data['date'])
data[['film_code', 'cinema_code']] = data[['film_code', 'cinema_code']].astype(str)

### Side bar   
with st.sidebar:
    
    st.sidebar.subheader('Dashboard parameter')
    month_filter = st.selectbox(label = 'Choose month', options = np.sort(data['month'].unique()).tolist())
    month_df = data[data['month'] == month_filter]    
    
    st.sidebar.subheader('Line chart parameter')
    chart1_type = st.selectbox(label = 'Choose type', options = ['line', 'scatter'])
        
    st.sidebar.subheader('Bar chart parameter')
    size = len(month_df['film_code'].unique()) + 1
    top_n_film = st.selectbox(label = 'Choose top film', options = range(1, size + 1))
    
    st.subheader('Specify plot height')
    plot_height = st.sidebar.slider('Specify plot height', 350, 500, 400)
    
### Title

st.markdown(
    """
    <style>
    .title {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(f'<h1 class="title">BÁO CÁO RẠP PHIM THÁNG {month_filter}</h1>', unsafe_allow_html=True)
# st.title(f"BÁO CÁO RẠP PHIM THÁNG {month_filter}")


# Row 1

# First plot
temp = month_df.groupby(['date']).agg({'total_sales': ['sum']}).sort_index().reset_index()
temp.columns = ['date', 'total_sales']
temp['date'] = pd.to_datetime(temp['date'])

if chart1_type == 'scatter':
    temp['peak'] = 0
    for i in temp['date'].values:
        i = pd.to_datetime(i)
        batch = temp[(temp['date'] > i - pd.DateOffset(days = 30)) & (temp['date'] < i + pd.DateOffset(days = 30))]
        if temp[temp['date'] == i]['total_sales'].values == max(batch['total_sales'].values):
            temp.loc[temp['date'] == i, 'peak'] += 1

    temp.loc[temp['peak'] == 0, 'color'] = 'blue'
    temp.loc[temp['peak'] == 1, 'color'] = 'red'

    fig = px.scatter(temp, x = 'date', y = 'total_sales')
    fig.update_traces(marker_color = temp['color'])

elif chart1_type == 'line':
    fig = px.line(temp, x = 'date', y = 'total_sales')
    
fig.update_layout(
    xaxis_title = 'Thời gian',
    yaxis_title = 'Tổng doanh thu',
    height = plot_height,
    title={
        'text': 'Tổng doanh thu theo ngày',
        'x': 0.5,  # Giữa trục x
        'xanchor': 'center',  # Căn giữa theo trục x
        'yanchor': 'top'  # Căn theo trục y
    }
)

col1, col2 = st.columns((1, 1))

with col1:
    st.plotly_chart(fig, use_container_width = True)


### Second chart
sc = month_df.groupby('film_code')['total_sales'].sum().reset_index()
sc = sc.sort_values('total_sales', ascending=False).head(top_n_film)

fig = px.bar(sc, x=range(top_n_film), y='total_sales')

fig.update_xaxes(
    tickmode='array',
    tickvals = list(range(top_n_film)),
    ticktext=sc['film_code']
)

fig.update_layout(
    xaxis_title = 'Mã phim',
    yaxis_title = 'Tổng doanh thu',
    height = plot_height,
    title={
        'text': f'Top {top_n_film} phim có doanh thu cao nhất trong tháng',
        'x': 0.5,  # Giữa trục x
        'xanchor': 'center',  # Căn giữa theo trục x
        'yanchor': 'top'  # Căn theo trục y
    }
)

with col2:
    # st.markdown(sc['film_code'])
    st.plotly_chart(fig, use_container_width = True)
    

    