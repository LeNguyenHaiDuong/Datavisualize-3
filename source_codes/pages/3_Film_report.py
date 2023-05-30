import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title = "Báo cáo doanh thu phim", layout = 'wide', 
                   initial_sidebar_state = 'expanded')

#Read data from csv
data = pd.read_csv('../datasets/data_preprocess.csv', index_col=0)

### Change type
data['date'] = pd.to_datetime(data['date'])
data[['film_code', 'cinema_code']] = data[['film_code', 'cinema_code']].astype(str)

### Side bar   
with st.sidebar:

    film_filter = st.selectbox(label = 'Chọn mã phim', options = np.sort(data['film_code'].unique())[:].tolist())
    film_df = data[data['film_code'] == film_filter]    

    
    month_filter = st.selectbox(label = 'Chọn tháng', options = np.sort(film_df['month'].unique())[:].tolist())
    month_df = film_df[film_df['month'] == month_filter]    

    max_cinemas = len(month_df.groupby('cinema_code'))
    if max_cinemas > 20:
        max_cinemas = 20
    if max_cinemas < 10:
        cinema_filter = st.slider('Chọn số lượng rạp:', 1, max_cinemas, max_cinemas)
    else:
        cinema_filter = st.slider('Chọn số lượng rạp:', 1, max_cinemas, 10)
    
    top_cinema =  month_df.groupby('cinema_code')['tickets_sold'].sum().sort_values(ascending=False)
    



    
    
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

st.markdown(f'<h1 class="title">BÁO CÁO PHIM {film_filter}</h1>', unsafe_allow_html=True)
st.markdown("#")


#st.write("Tháng: ", month_filter)

### Merics rows
#=========================================================
col1, col2, col3, col4 = st.columns([2, 4, 3, 2])

### First metrict: Rank
sales_by_film = data[data["month"] == month_filter].groupby("film_code")["total_sales"].sum().sort_values(ascending=False)
film_rank = sales_by_film.index.get_loc(film_filter) + 1

with col1:
    st.metric("Rank", '#{:,.0f}'.format(film_rank))

### Second metric: Doanh thu
### Doanh thu:
total_profit = int(month_df['total_sales'].sum())
pre_total_profit = int(film_df[film_df['month'] == month_filter - 1]['total_sales'].sum())
if pre_total_profit == 0:  
    raise_ = 0
else: 
    raise_ = (total_profit - pre_total_profit) / pre_total_profit
with col2:
    st.metric("Tổng doanh thu", '{:,.0f}'.format(total_profit), str(round(raise_, 2)) + '%')


### Third metric: Số xuất chiếu
total_showtime = int(month_df['show_time'].sum())
pre_total_showtime = int(film_df[film_df['month'] == month_filter - 1]['show_time'].sum())
if pre_total_showtime == 0:
    raise_ = 0
else:
    raise_ = total_showtime - pre_total_showtime
with col3:
    st.metric("Tổng số suất chiếu", '{:,.0f}'.format(total_showtime), raise_)


# Forth metric: Số vé bán được
total_tickets = int(month_df['tickets_sold'].sum())
pre_total_tickets = int(film_df[film_df['month'] == month_filter - 1]['tickets_sold'].sum())
if pre_total_tickets == 0:
    raise_ = 0
else:
    raise_ = (total_tickets - pre_total_tickets) / pre_total_tickets
with col4:
    st.metric("Tổng số vé bán ra", '{:,.0f}'.format(total_tickets), str(round(raise_, 2)) + '%')
#=================================================================


### Row 1
sales_by_day = month_df.groupby('day')['total_sales'].sum()
fig = px.line(sales_by_day, x = sales_by_day.index, y = sales_by_day.values, title="Doanh thu theo ngày")

fig.update_layout(
    xaxis_title = 'Ngày',
    yaxis_title = 'Tổng doanh thu',
    title={
        'text': 'Doanh thu của các ngày trong tháng',
        'x': 0.5,  # Giữa trục x
        'xanchor': 'center',  # Căn giữa theo trục x
        'yanchor': 'top'  # Căn theo trục y
    }
)


st.plotly_chart(fig, use_container_width = True)
#====================================================================
#### Row 2:
col1, col2 = st.columns([5, 5])
## First chart
tickets_by_day = month_df.groupby('day')['tickets_sold'].sum()
fig = px.line(tickets_by_day, x = tickets_by_day.index, y = tickets_by_day.values, title="Số vé bán theo ngày")

fig.update_layout(
    xaxis_title = 'Ngày',
    yaxis_title = 'Tổng vé bán',
    title={
        'text': 'Số vé bán của các ngày trong tháng',
        'x': 0.5,  # Giữa trục x
        'xanchor': 'center',  # Căn giữa theo trục x
        'yanchor': 'top'  # Căn theo trục y
    }
)

with col1:
    st.plotly_chart(fig, use_container_width=True)

## Second chart
show_time_by_day = month_df.groupby('day')['show_time'].sum()
fig = px.line(show_time_by_day, x = show_time_by_day.index, y = show_time_by_day.values, title="Số suất chiếu theo ngày")

fig.update_layout(
    xaxis_title = 'Ngày',
    yaxis_title = 'Số suất chiếu',
    title={
        'text': 'Số suất chiếu của các ngày trong tháng',
        'x': 0.5,  # Giữa trục x
        'xanchor': 'center',  # Căn giữa theo trục x
        'yanchor': 'top'  # Căn theo trục y
    }
)

with col2:
    st.plotly_chart(fig, use_container_width=True)




#### Row 3
##### First chart:
col1, col2 = st.columns((3.5, 6.5))
bins = [0, 50000, 80000, 100000, 120000, float('inf')]

# Chia cột "ticket_price" thành bins
month_df['price_bin'] = pd.cut(month_df['ticket_price'], bins=bins, labels=False, right=False)

# Đổi tên các bins thành các giá trị tương ứng
bin_labels = ['price <= 50000', '50000 < price <= 80000', '80000 < price <= 100000', '100000 < price <= 120000', 'price > 120000']
month_df['price_bin'] = month_df['price_bin'].replace(range(len(bin_labels)), bin_labels)

# Tính tổng số lượng phần tử trong mỗi bin
bin_counts = month_df['price_bin'].value_counts().reset_index()

# Tạo biểu đồ pie chart bằng Plotly
fig = px.pie(bin_counts, values='price_bin', names='index', title='Phân bố giá')
#fig.update_traces(hoverinfo='label+percent', textinfo='value+percent')

fig.update_layout(
    title={
        'text': 'Phân bố giá  vé của phim tại các rạp',
        'x': 0.5,  # Giữa trục x
        'xanchor': 'center',  # Căn giữa theo trục x
        'yanchor': 'top'  # Căn theo trục y
    }

)

with col1:
    st.plotly_chart(fig, use_container_width=True)
#### Second chart:
month_df['price_bin'] = pd.Categorical(month_df['price_bin'], categories=bin_labels, ordered=True)

tickets_by_price = month_df.groupby('price_bin')['tickets_sold'].sum().reset_index()

fig2 = px.bar(tickets_by_price, x='price_bin', y='tickets_sold', title='Phân bố số lượng vé bán ra theo giá vé')

fig2.update_layout(
    xaxis_title = 'Giá vé',
    yaxis_title = 'Tổng vé bán',
    title={
        'text': 'Số lượng vé bán theo giá',
        'x': 0.5,  # Giữa trục x
        'xanchor': 'center',  # Căn giữa theo trục x
        'yanchor': 'top'  # Căn theo trục y
    }
)

with col2:
    st.plotly_chart(fig2, use_container_width=True)

## Row 4:
col1, col2 = st.columns((3, 1))

top_x_cinema = top_cinema.iloc[:cinema_filter].sort_values()


fig = px.bar(top_x_cinema, x=top_x_cinema.values, y=top_x_cinema.index, orientation='h')

fig.update_layout(
    xaxis_title = 'Số vé bán',
    yaxis_title = 'Mã rạp phim',
    yaxis={'type': 'category'} ,
    title={
        'text': 'Số lượng vé bán ở các rạp phim',
        'x': 0.5,  # Giữa trục x
        'xanchor': 'center',  # Căn giữa theo trục x
        'yanchor': 'top'  # Căn theo trục y
    },
    #width = 300
)
with col2:
    st.plotly_chart(fig, use_container_width=True)

## Second chart
cinema_df = month_df[month_df['cinema_code'].isin(top_x_cinema.index)]
cinema_df = cinema_df.groupby(["cinema_code", "day"])["occu_perc"].mean().unstack(level="day")


fig = go.Figure(data=go.Heatmap(
    z=cinema_df.values,
    x=cinema_df.columns,
    y=cinema_df.index,
    colorscale='Viridis'
))

fig.update_layout(
    title={
        'text': 'Tỷ lệ chiếm chỗ tại các rạp',
        'x': 0.5,  # Giữa trục x
        'xanchor': 'center',  # Căn giữa theo trục x
        'yanchor': 'top'  # Căn theo trục y
    },
    xaxis_title='Day',
    yaxis_title='Cinema Code',
    yaxis={'type': 'category'} ,
   # width = 1200
)

with col1:
    st.plotly_chart(fig, use_container_width=True)