import csv
import random
from datetime import datetime, timedelta

# Configuration
num_records = 1500
cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami']
sub_types = ['Basic', 'Standard', 'Premium']
genders = ['Male', 'Female', 'Other']
devices = ['Mobile', 'Desktop', 'Tablet']

# Base dates
start_date = datetime(2026, 1, 1)

with open('customer_churn_dataset.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Header row
    writer.writerow([
        'Customer ID', 'Age', 'Gender', 'City', 'Subscription Type', 
        'Monthly Spending', 'Tenure', 'Number of Purchases', 
        'Customer Support Requests', 'Login Frequency', 'Last Activity Date', 
        'Satisfaction Score', 'Device Type', 'Total Spending', 'Churn Status'
    ])
    
    for i in range(1, num_records + 1):
        cust_id = f"CUST-{1000 + i}"
        age = random.randint(18, 70)
        gender = random.choice(genders)
        city = random.choice(cities)
        sub_type = random.choice(sub_types)
        
        # Set realistic constraints based on sub type
        if sub_type == 'Basic':
            monthly_spending = round(random.uniform(25.00, 45.00), 2)
        elif sub_type == 'Standard':
            monthly_spending = round(random.uniform(46.00, 75.00), 2)
        else:
            monthly_spending = round(random.uniform(76.00, 120.00), 2)
            
        tenure = random.randint(1, 48)  # months
        num_purchases = random.randint(1, max(2, tenure // 2))
        
        # Logic to make churn correlations realistic
        # Higher support requests + low satisfaction = higher churn risk
        support_requests = random.randint(0, 7)
        satisfaction_score = random.randint(1, 5)
        
        if support_requests >= 4 or satisfaction_score <= 2:
            churn_status = random.choices([0, 1], weights=[30, 70])[0]
        else:
            churn_status = random.choices([0, 1], weights=[90, 10])[0]
            
        login_frequency = random.randint(1, 30) if churn_status == 0 else random.randint(1, 12)
        
        # Calculate random date within the last 6 months
        days_ago = random.randint(0, 30) if churn_status == 0 else random.randint(10, 120)
        last_activity_date = (datetime(2026, 6, 25) - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        device = random.choice(devices)
        total_spending = round(monthly_spending * tenure, 2)
        
        # Write row
        writer.writerow([
            cust_id, age, gender, city, sub_type, monthly_spending, 
            tenure, num_purchases, support_requests, login_frequency, 
            last_activity_date, satisfaction_score, device, total_spending, churn_status
        ])

print(f"Success! Created 'customer_churn_dataset.csv' with {num_records} rows.")