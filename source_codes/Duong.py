import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import datetime 
import plotly.graph_objects as go

st.set_page_config(page_title = "Doanh thu hang thang cua cac rap chieu phim", layout = 'wide', 
                   initial_sidebar_state = 'expanded')

data = pd.read_csv("../datasets/data_preprocess.csv", index_col = 0)

### Change type
data['date'] = pd.to_datetime(data['date'])
data[['film_code', 'cinema_code']] = data[['film_code', 'cinema_code']].astype(str)

### Side bar   
with st.sidebar:
    
    st.sidebar.subheader('Báo cáo tháng')
    month_filter = st.selectbox(label = 'Choose month', options = np.sort(data['month'].unique())[1:].tolist())
    month_df = data[data['month'] == month_filter]    
    
    st.sidebar.subheader('Top phim trong tháng')
    size = len(month_df['film_code'].unique())
    top_n_film = st.selectbox(label = 'Choose top film', options = range(1, size + 1))
    
    st.sidebar.subheader('Top rạp trong tháng')
    size = len(month_df['cinema_code'].unique())
    top_n_cinema = st.selectbox(label = 'Choose top cinema', options = range(1, size + 1))
    
    st.subheader('Độ cao hàng 1')
    plot1_height = st.sidebar.slider('Specify first row height', 350, 500, 400)
    
    st.sidebar.subheader('Donut Chart')
    size = len(month_df['film_code'].unique())
    n_film = st.selectbox(label = 'Choose num film', options = range(1, min(5, size + 1)))
    
    st.sidebar.subheader('Top rạp trong tháng')
    size = len(month_df['cinema_code'].unique())
    n_cinema = st.selectbox(label = 'Choose num cinema', options = range(1, min(5, size + 1)))
    
    st.sidebar.subheader('Doanh thu theo ngày')
    chart1_type = st.selectbox(label = 'Choose type', options = ['line', 'scatter'])
    
    st.subheader('Độ cao hàng 3')
    plot2_height = st.sidebar.slider('Specify second row height', 350, 500, 400)
    
    st.subheader('Mô hình ARIMA')
    data['date'] = pd.to_datetime(data['date'])
    start_date = data['date'].min().to_pydatetime()
    end_date = data['date'].max().to_pydatetime()
    to = st.slider('Choose end date for train', start_date, end_date, value = end_date)
    
    st.subheader('Số ngày muốn dự đoán')
    num_day = st.sidebar.slider('Choose num of day', 5, 10, 15)
    
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

### Row 1
col1, col2, col3, col4 = st.columns([3.5, 1.5, 2, 2])

### First metric: Doanh thu
### Doanh thu:
total_profit = int(month_df['total_sales'].sum())
pre_total_profit = int(data[data['month'] == month_filter - 1]['total_sales'].sum())
raise_ = (total_profit - pre_total_profit) / pre_total_profit
with col1:
    st.metric("Tổng doanh thu", '{:,.0f}'.format(total_profit), str(round(raise_, 2)) + '%')

### Second metric: Thống kê số phim
total_film = len(month_df['film_code'].unique())
pre_total_film = len(data[data['month'] == month_filter - 1]['film_code'].unique())
raise_ = total_film - pre_total_film
with col2:
    st.metric("Tổng số phim", '{:,.0f}'.format(total_film), raise_)

### Third metric: Số xuất chiếu
total_showtime = len(month_df.index)
pre_total_showtime = len(data[data['month'] == month_filter - 1].index)
raise_ = total_showtime - pre_total_showtime
with col3:
    st.metric("Tổng số suất chiếu", '{:,.0f}'.format(total_showtime), raise_)


# Forth metric: Số vé bán được
total_tickets = int(month_df['tickets_sold'].sum())
pre_total_tickets = int(data[data['month'] == month_filter - 1]['tickets_sold'].sum())
raise_ = (total_tickets - pre_total_tickets) / pre_total_tickets
with col4:
    st.metric("Tổng số vé bán ra", '{:,.0f}'.format(total_tickets), str(round(raise_, 2)) + '%')


### Row 1
col1, col2 = st.columns((1, 1))

### First chart
sc1 = month_df.groupby('film_code')['total_sales'].sum().reset_index()
sc1 = sc1.sort_values('total_sales', ascending=False)
sc = sc1.head(top_n_film)

fig = px.bar(sc, x=range(top_n_film), y='total_sales')

fig.update_xaxes(
    tickmode='array',
    tickvals = list(range(top_n_film)),
    ticktext = sc['film_code']
)

fig.update_layout(
    xaxis_title = 'Mã phim',
    yaxis_title = 'Tổng doanh thu',
    height = plot1_height,
    title={
        'text': f'Top {top_n_film} phim có doanh thu cao nhất trong tháng',
        'x': 0.5,  # Giữa trục x
        'xanchor': 'center',  # Căn giữa theo trục x
        'yanchor': 'top'  # Căn theo trục y
    }
)

with col1:
    st.plotly_chart(fig, use_container_width = True)
    
### Second chart
sc2 = month_df.groupby('cinema_code')['total_sales'].sum().reset_index()
sc2 = sc2.sort_values('total_sales', ascending=False)
sc = sc2.head(top_n_cinema)

fig = px.bar(sc, x=range(top_n_cinema), y='total_sales')

fig.update_xaxes(
    tickmode='array',
    tickvals = list(range(top_n_cinema)),
    ticktext = sc['cinema_code']
)

fig.update_layout(
    xaxis_title = 'Mã rạp',
    yaxis_title = 'Tổng doanh thu',
    height = plot1_height,
    title={
        'text': f'Top {top_n_cinema} rạp có doanh thu cao nhất trong tháng',
        'x': 0.5,  # Giữa trục x
        'xanchor': 'center',  # Căn giữa theo trục x
        'yanchor': 'top'  # Căn theo trục y
    }
)

with col2:
    st.plotly_chart(fig, use_container_width = True)

### Row 2
col1, col2 = st.columns((1, 1))


temp = month_df.copy()
temp['weekday'] = data['date'].dt.day_name()
temp = temp.groupby([temp['date'].dt.day_of_week, temp['weekday']]).agg({'total_sales': 'mean'}).reset_index()
fig = px.bar(x = temp['weekday'], y = temp['total_sales'])
fig.update_layout(
    xaxis_title = 'Ngày trong tuần',
    yaxis_title = 'Doanh thu trung bình',
    height = plot2_height,
    title={
        'text': 'Doanh thu trung bình của các ngày trong tuần',
        'x': 0.5,  # Giữa trục x
        'xanchor': 'center',  # Căn giữa theo trục x
        'yanchor': 'top'  # Căn theo trục y
    }
)

with col1:
    st.plotly_chart(fig, use_container_width = True)


def top_n_and_other(df, n):
    new_df = df.head(n).T
    new_df['new'] = pd.Series(['other'])
    new_df.at['total_sales', 'new'] = df.iloc[n:]['total_sales'].sum()
    try:
        new_df.at['film_code', 'new'] = 'other'
    except:
        new_df.at['cinema_code', 'new'] = 'other'
    st.dataframe(new_df.T)
    return new_df.T
         

fig = go.Figure()

new_df_film = top_n_and_other(sc2, n_film)
fig.add_trace(go.Pie(labels = new_df_film['cinema_code'],
                             values = new_df_film['total_sales'],
                             hole = 0.4)
              )

new_df_cinema = top_n_and_other(sc1, n_cinema)
fig.add_trace(go.Pie(labels = new_df_cinema['film_code'],
                             values = new_df_cinema['total_sales'],
                             hole = 0.8)
              )

fig.update_layout(
    title={
        'text': 'Doanh thu trung bình của các ngày trong tuần',
        'x': 0.5,  # Giữa trục x
        'xanchor': 'center',  # Căn giữa theo trục x
        'yanchor': 'top'  # Căn theo trục y
    }
)

with col2:
    st.plotly_chart(fig, use_container_width = True)

### Row 3
col = st.columns((1))

# First plot
temp = month_df.groupby(['date']).agg({'total_sales': ['sum']}).sort_index().reset_index()
temp.columns = ['date', 'total_sales']
temp['date'] = pd.to_datetime(temp['date'])

if chart1_type == 'scatter':
    temp['peak'] = 0
    for i in temp['date'].values:
        i = pd.to_datetime(i)
        batch = temp[(temp['date'] > i - pd.DateOffset(days = 10)) & (temp['date'] < i + pd.DateOffset(days = 10))]
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
    height = plot2_height,
    title={
        'text': 'Tổng doanh thu theo ngày',
        'x': 0.5,  # Giữa trục x
        'xanchor': 'center',  # Căn giữa theo trục x
        'yanchor': 'top'  # Căn theo trục y
    }
)

with col[0]:
    st.plotly_chart(fig, use_container_width = True)

### Row 4
st.markdown(f'<h1 class="title">Mô hình ARIMA dự đoán doanh thu</h1>', unsafe_allow_html=True)

### Row 5
col1, col2 = st.columns((3, 1))

### ARIMA
temp = data[data['date'] <= to]
train = temp[['date', 'total_sales']]
train = train.groupby(['date'])['total_sales'].sum()
train.columns = ['total_sales']
fill_data = pd.DataFrame(index = ['total_sales'])

i = train.index[0]
while i <= to:
    if i not in train.index:
        prev = i - pd.DateOffset(days=1)
        while prev not in train.index:
            prev = prev - pd.DateOffset(days=1)
            
        next_ = i + pd.DateOffset(days=1)
        while next_ not in train.index:
            next_ = next_ + pd.DateOffset(days=1)
        
        val = (train.loc[prev] * (next_ - i).days + train.loc[next_] * (i - prev).days) / (next_ - prev).days
        fill_data[i.strftime('%Y-%m-%d')] = round(val)
    i += pd.DateOffset(days=1)
fill_data = fill_data.T
train = pd.concat([train, fill_data])
train.index = pd.to_datetime(train.index)
train['total_sales'] = train['total_sales'].fillna(train[0])
train = train.drop(0, axis=1)
train = train.sort_index()

model = ARIMA(train['total_sales'], order=(3, 1, 2))
model_fit = model.fit()

from_ = train.index[-1]
to_ =  train.index[-1] + pd.DateOffset(days = num_day)
predictions = model_fit.predict(start = from_, end = to_)

fig = go.Figure()
fig.add_trace(go.Scatter(x = train.index, y = train['total_sales'], name='Train data', mode = 'lines'))
fig.add_trace(go.Scatter(x = predictions.index, y = predictions.values, name = 'Prediction', mode = 'lines', line=dict(color='red')))

fig.update_layout(
    xaxis_title = 'Thời gian',
    yaxis_title = 'Tổng doanh thu',
    height = plot2_height,
    title={
        'text': f"Dự đoán doanh thu từ {from_.strftime('%Y-%m-%d')} đến {to_.strftime('%Y-%m-%d')}",
        'x': 0.5,  # Giữa trục x
        'xanchor': 'center',  # Căn giữa theo trục x
        'yanchor': 'top'  # Căn theo trục y
    },
    legend=dict(
        x = 0.9,  # Vị trí ngang, giữa trục x
        y = 1.1,  # Vị trí dọc, ở trên cùng
        xanchor='center',  # Căn giữa theo trục x
        yanchor='top'  # Căn theo trục y
    )
)

with col1:
    st.plotly_chart(fig, use_container_width = True)
    
with col2:
    st.dataframe(predictions)
    