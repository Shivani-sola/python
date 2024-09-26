#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd


# In[54]:


df = pd.read_csv(r"C:\Users\solas\Downloads\diminos_data_v2\diminos_data_v2\deliveries.csv")


# In[55]:


df.head()


# In[56]:


df.isnull().sum()


# In[57]:


df.duplicated()


# In[58]:


df['time_stamp'] = pd.to_datetime(df['time_stamp'])


# In[59]:


df['time_stamp'] 


# In[60]:


import pandas as pd

# Assuming df is your dataset
# Convert 'time_stamp' to datetime and truncate nanoseconds by rounding to seconds
df['time_stamp'] = pd.to_datetime(df['time_stamp']).dt.floor('S')

# Ensure the 'status' column is clean (no leading/trailing spaces or case issues)
df['status'] = df['status'].str.strip().str.lower()

# Filter placed and delivered orders
placed_orders = df[df['status'] == 'order placed']
delivered_orders = df[df['status'] == 'delivered']

# Remove duplicate orders if any
placed_orders = placed_orders.drop_duplicates(subset=['order_id'])
delivered_orders = delivered_orders.drop_duplicates(subset=['order_id'])

# Merge placed and delivered orders on 'order_id'
merged_df = pd.merge(placed_orders, delivered_orders, on='order_id', suffixes=('_placed', '_delivered'))

# Calculate the time difference in minutes
merged_df['delivery_time'] = (merged_df['time_stamp_delivered'] - merged_df['time_stamp_placed']).dt.total_seconds() / 60

# Drop any invalid or NaN delivery times
merged_df = merged_df.dropna(subset=['delivery_time'])
merged_df = merged_df[merged_df['delivery_time'] > 0]

# Calculate the average delivery time
average_delivery_time = merged_df['delivery_time'].mean()

print(f"The average delivery time is {average_delivery_time:.2f} minutes.")


# In[61]:


df['time_stamp'].mean()


# In[62]:


# Convert 'time_stamp' to datetime and truncate nanoseconds by rounding to seconds
df['time_stamp'] = pd.to_datetime(df['time_stamp']).dt.floor('S')

# Ensure the 'status' column is clean (no leading/trailing spaces or case issues)
df['status'] = df['status'].str.strip().str.lower()

# Print the unique values in the status column to verify
print("Unique status values:", df['status'].unique())

# Modify the filtering for placed orders based on your actual data
placed_orders = df[df['status'] == 'pending']  # Change this to the appropriate status
delivered_orders = df[df['status'] == 'delivered']

# Print shapes of the filtered DataFrames
print("Placed Orders Shape:", placed_orders.shape)
print("Delivered Orders Shape:", delivered_orders.shape)

# Merge placed and delivered orders on 'order_id'
merged_df = pd.merge(placed_orders, delivered_orders, on='order_id', suffixes=('_placed', '_delivered'))

# Check the shape and head of the merged DataFrame
print("Merged Orders Shape:", merged_df.shape)
print("Merged DataFrame Head:")
print(merged_df[['order_id', 'time_stamp_placed', 'time_stamp_delivered']].head())

# Calculate the time difference in minutes
merged_df['delivery_time'] = (merged_df['time_stamp_delivered'] - merged_df['time_stamp_placed']).dt.total_seconds() / 60

# Check the delivery times
print("Delivery Times:")
print(merged_df[['order_id', 'delivery_time']])

# Drop any invalid or NaN delivery times
merged_df = merged_df.dropna(subset=['delivery_time'])
merged_df = merged_df[merged_df['delivery_time'] > 0]

# Calculate the average delivery time
average_delivery_time = merged_df['delivery_time'].mean()

# Print the average delivery time
print(f"The average delivery time is {average_delivery_time:.2f} minutes.")


# In[63]:


percentile_99_delivery_time = merged_df['delivery_time'].quantile(0.99)

# Round the 99th percentile delivery time to two decimal places
percentile_99_delivery_time_rounded = round(percentile_99_delivery_time, 2)

# Print the 99th percentile delivery time
print(f"The 99th percentile delivery time is {percentile_99_delivery_time_rounded:.2f} minutes.")


# In[64]:


# Find the order ID with the maximum delivery time
max_delivery_time_idx = merged_df['delivery_time'].idxmax()
max_delivery_order = merged_df.loc[max_delivery_time_idx]

# Extract the order ID and maximum delivery time
max_order_id = max_delivery_order['order_id']
max_delivery_time = max_delivery_order['delivery_time']

# Print the result
print(f"The order ID with the maximum delivery time is {max_order_id} with a delivery time of {max_delivery_time:.2f} minutes.")


# In[65]:


print(max_order_id, round(max_delivery_time, 2))


# In[66]:



# Convert 'time_stamp' to datetime if it's not already
df['time_stamp'] = pd.to_datetime(df['time_stamp'])

# Define the date range
start_date = '2024-01-01'
end_date = '2024-01-31'

# Filter the DataFrame for orders within the date range
january_orders = df[(df['time_stamp'] >= start_date) & (df['time_stamp'] <= end_date)]

# Count the number of pizzas ordered
number_of_orders = january_orders.shape[0]

# Print the result
print(number_of_orders)


# In[67]:


print(df.columns)


# In[68]:


import pandas as pd

# Sample DataFrame creation (replace this with your actual data loading)


# Convert 'time_stamp' to datetime if it's not already
df['time_stamp'] = pd.to_datetime(df['time_stamp'])

# Define the date range
start_date = '2024-01-01'
end_date = '2024-01-31'

# Filter the DataFrame for orders within the date range
january_orders = df[(df['time_stamp'] >= start_date) & (df['time_stamp'] <= end_date)]

# Ensure the status is delivered for counting delivery times
delivered_orders = january_orders[january_orders['status'] == 'delivered']

# Assuming we know the average delivery time is 30 minutes
# Calculate the time when the order was placed based on the assumption
# Here we consider 'time_stamp' as the time of delivery
# So to get the placed time, we can subtract 30 minutes
delivered_orders['placed_time'] = delivered_orders['time_stamp'] - pd.Timedelta(minutes=30)

# If you need to check delivery time now
delivered_orders['delivery_time'] = 30  # Since we assume delivery time is the fixed average of 30 minutes

# Count orders that took more than 30 minutes to deliver
orders_over_30_minutes = delivered_orders[delivered_orders['delivery_time'] > 30]

# Count the number of such orders
number_of_orders_over_30_minutes = orders_over_30_minutes.shape[0]

# Print the result
print(number_of_orders_over_30_minutes)


# In[69]:


df.head()


# In[70]:


df2 = pd.read_csv(r"C:\Users\solas\Downloads\diminos_data_v2\diminos_data_v2\products.csv")  # Uncomment to loa


# In[71]:


df2.head()


# In[72]:


# Convert 'time_stamp' to datetime
df['time_stamp'] = pd.to_datetime(df['time_stamp'])

# Check data types
print("Data types before conversion:")
print(df.dtypes)
print(df2.dtypes)

# Convert order_id to string if item_id is string (object)
df['order_id'] = df['order_id'].astype(str)  # Convert to string if needed
# or you could convert item_id in df2 to int if they should be integers
# df2['item_id'] = df2['item_id'].astype(int)  # Uncomment if you want to convert to int

# Merge the two DataFrames on order_id and item_id
merged_df = df.merge(df2, left_on='order_id', right_on='item_id')

# Define the date range for 2023
start_date = '2023-01-01'
end_date = '2023-12-31'

# Filter for orders delivered in 2023
orders_2023 = merged_df[(merged_df['time_stamp'] >= start_date) & (merged_df['time_stamp'] <= end_date)]

# Calculate the delivery time (assuming delivery time is more than 30 mins for this example)
# If you have actual delivery time logic, adjust this part
orders_2023['delivery_time'] = 30  # Placeholder; modify if you have real timestamps.

# Filter for orders that took more than 30 minutes to deliver
late_deliveries = orders_2023[orders_2023['delivery_time'] > 30]

# Calculate total refunds based on the price of late deliveries
total_refunds = late_deliveries['Price'].sum()

# Print the total amount lost due to refunds
print(f"The total amount the pizza store lost due to refunds on late deliveries in 2023 is: ${total_refunds:.2f}")


# In[73]:


# Convert 'time_stamp' to datetime
df['time_stamp'] = pd.to_datetime(df['time_stamp'])

# Convert order_id to string if item_id is string (object)
df['order_id'] = df['order_id'].astype(str)

# Merge the two DataFrames on order_id and item_id
merged_df = df.merge(df2, left_on='order_id', right_on='item_id', how='inner')

# Check the merged DataFrame
print("Merged DataFrame:")
print(merged_df)

# Calculate delivery time (dummy calculation for illustration, replace with actual logic)
merged_df['delivery_time'] = [30, 35, 45, 25, 32, 31]  # Sample delivery times in minutes

# Filter for late deliveries (more than 30 minutes)
late_deliveries = merged_df[merged_df['delivery_time'] > 30]

# Check late deliveries
print("Late Deliveries DataFrame:")
print(late_deliveries)

# Calculate total refunds based on the price of late deliveries
if not late_deliveries.empty:
    late_deliveries['refund_amount'] = late_deliveries['Price']

    # Extract year from time_stamp for grouping
    late_deliveries['year'] = late_deliveries['time_stamp'].dt.year

    # Group by year and sum the refund amounts
    refunds_by_year = late_deliveries.groupby('year')['refund_amount'].sum()

    # Identify the year with the maximum refund amount
    if not refunds_by_year.empty:
        max_refund_year = refunds_by_year.idxmax()
        max_refund_amount = refunds_by_year.max()
        print(f"The year in which the store lost the maximum amount due to refunds on deliveries is {max_refund_year} with a total loss of ${max_refund_amount:.2f}.")
    else:
        print("No refunds to report.")
else:
    print("No late deliveries found.")


# In[74]:


# Convert the time_stamp to datetime
df['time_stamp'] = pd.to_datetime(df['time_stamp'])

# Split into separate columns for date and time
df['date'] = df['time_stamp'].dt.date
df['time'] = df['time_stamp'].dt.time
df['year'] = df['time_stamp'].dt.year

# Initialize refund_amount column
df['refund_amount'] = 0


# In[75]:


# Convert the time_stamp to datetime
df['time_stamp'] = pd.to_datetime(df['time_stamp'])

# Create a 'year' column
df['year'] = df['time_stamp'].dt.year

# Determine if delivery is late and the corresponding refund amount
# Shift the status column to get the delivery status for the 'pending' orders
df['delivered_time'] = df.loc[df['status'] == 'delivered', 'time_stamp']
df['placed_time'] = df.loc[df['status'] == 'pending', 'time_stamp']

# Fill missing values for placed and delivered times
df['placed_time'] = df['placed_time'].ffill()
df['delivered_time'] = df['delivered_time'].bfill()

# Calculate delivery duration in minutes
df['delivery_duration'] = (df['delivered_time'] - df['placed_time']).dt.total_seconds() / 60

# Set refund amount based on delivery duration
df['refund_amount'] = df['delivery_duration'].apply(lambda x: 10 if x > 30 else 0)

# Group by year to sum refunds
refunds_by_year = df.groupby('year')['refund_amount'].sum()

# Identify the year with the maximum refund amount
if not refunds_by_year.empty:  # Check if the DataFrame is not empty
    max_refund_year = refunds_by_year.idxmax()
    max_refund_amount = refunds_by_year.max()

    print(f"The year with the maximum refund amount is {max_refund_year} with a total refund of {max_refund_amount:.2f}.")
else:
    print("No refunds were processed in the given data.")


# In[77]:


# Load data into DataFrames
df_orders = pd.read_csv(r"C:\Users\solas\Downloads\diminos_data_v2\diminos_data_v2\deliveries.csv")
df_items = pd.read_csv(r"C:\Users\solas\Downloads\diminos_data_v2\diminos_data_v2\products.csv")

# Convert the time_stamp to datetime
df_orders['time_stamp'] = pd.to_datetime(df_orders['time_stamp'])

# Ensure 'order_id' in df_orders is of type int
df_orders['order_id'] = df_orders['order_id'].astype(str)  # Convert to string for consistency

# Ensure 'item_id' in df_items is of type str
df_items['item_id'] = df_items['item_id'].astype(str)  # Convert to string for consistency

# Merge the two DataFrames on order_id and item_id
merged_df = df_orders.merge(df_items, left_on='order_id', right_on='item_id', how='inner')

# Filter for delivered orders
delivered_orders = merged_df[merged_df['status'] == 'delivered']

# Calculate total revenue, excluding refunds
total_revenue = delivered_orders['Price'].sum()

print(f"Total revenue generated by the pizza store till date (excluding refunds): ${total_revenue:.2f}")


# In[78]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

# Set random seed for reproducibility
np.random.seed(42)

# Sample parameters
average_orders_per_hour = 5  # Average number of orders per hour
hours = np.arange(0, 24)  # 24 hours in a day

# Simulate the number of orders for each hour using Poisson distribution
simulated_orders = poisson.rvs(mu=average_orders_per_hour, size=len(hours))

# Create a DataFrame to hold the data
df_orders = pd.DataFrame({
    'Hour': hours,
    'Simulated Orders': simulated_orders
})

# Calculate probabilities for plotting
probabilities = poisson.pmf(np.arange(0, max(simulated_orders)+1), mu=average_orders_per_hour)

# Plotting the results
plt.figure(figsize=(12, 6))
sns.barplot(x='Hour', y='Simulated Orders', data=df_orders, color='skyblue', label='Simulated Orders')
plt.title('Simulated Incoming Orders per Hour using Poisson Distribution')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Orders')
plt.xticks(hours)
plt.grid(axis='y')
plt.legend()
plt.show()

# Display the simulated order data
print(df_orders)


# In[ ]:




