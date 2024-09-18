import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

# Define the fuzzy variables (inputs)
age = ctrl.Antecedent(np.arange(18, 81, 1), 'age')
crop_season = ctrl.Antecedent(np.arange(0, 3, 1), 'crop_season')  # 0-rabi(most profitable), 1-kharif, 2-Zaid 
guarantor = ctrl.Antecedent(np.arange(0, 2, 1), 'guarantor')  # 0 = no, 1 = yes
financial_records = ctrl.Antecedent(np.arange(0, 100, 1), 'financial_records')  # 0-100 trust level
farming_history = ctrl.Antecedent(np.arange(0, 100, 1), 'farming_history')  # 0-100 trust level
bank_statements = ctrl.Antecedent(np.arange(0, 100, 1), 'bank_statements')  # 0-100 trust level
inputs_required = ctrl.Antecedent(np.arange(0, 100, 1), 'inputs_required')  # Cost per acre (scaled)
market = ctrl.Antecedent(np.arange(0, 3, 1), 'market')  # 0=direct, 1=middleman, 2=processor
land_acreage = ctrl.Antecedent(np.arange(0, 6, 1), 'land_acreage')  # Size in acres (scaled)
rainfall = ctrl.Antecedent(np.arange(0, 500, 1), 'rainfall')  # Average rainfall (cm)
temperature = ctrl.Antecedent(np.arange(0, 50, 1), 'temperature')  # Average temperature (°C)
income_source = ctrl.Antecedent(np.arange(0, 100, 1), 'income_source')  # Stability of income (0-100)
annual_expenditure = ctrl.Antecedent(np.arange(0, 1000000, 1), 'annual_expenditure')  # Expenditure (scaled)
insurance = ctrl.Antecedent(np.arange(0, 2, 1), 'insurance')  # 0 = no insurance, 1 = insured

# Output: Risk (low, medium, high)
risk = ctrl.Consequent(np.arange(0, 10, 1), 'risk')

# Define membership functions for all inputs
age['young'] = fuzz.trimf(age.universe, [18, 18, 40])
age['middle_aged'] = fuzz.trimf(age.universe, [32, 50, 65])
age['old'] = fuzz.trimf(age.universe, [50, 80, 80])

crop_season['low'] = fuzz.trimf(crop_season.universe, [1, 2, 2])
crop_season['medium'] = fuzz.trimf(crop_season.universe, [0, 1, 1])
crop_season['high'] = fuzz.trimf(crop_season.universe, [0, 0, 0])

guarantor['no'] = fuzz.trimf(guarantor.universe, [0, 0, 1])
guarantor['yes'] = fuzz.trimf(guarantor.universe, [1, 1, 1])

financial_records['poor'] = fuzz.trimf(financial_records.universe, [0, 0, 50])
financial_records['average'] = fuzz.trimf(financial_records.universe, [35, 50, 70])
financial_records['good'] = fuzz.trimf(financial_records.universe, [60, 100, 100])

farming_history['poor'] = fuzz.trimf(farming_history.universe, [0, 0, 50])
farming_history['average'] = fuzz.trimf(farming_history.universe, [30, 50, 70])
farming_history['good'] = fuzz.trimf(farming_history.universe, [60, 100, 100])

bank_statements['poor'] = fuzz.trimf(bank_statements.universe, [0, 0, 50])
bank_statements['average'] = fuzz.trimf(bank_statements.universe, [30, 50, 70])
bank_statements['good'] = fuzz.trimf(bank_statements.universe, [60, 100, 100])

inputs_required['low'] = fuzz.trimf(inputs_required.universe, [0, 0, 50])
inputs_required['medium'] = fuzz.trimf(inputs_required.universe, [30, 50, 70])
inputs_required['high'] = fuzz.trimf(inputs_required.universe, [60, 100, 100])

market['direct'] = fuzz.trimf(market.universe, [0, 0, 1])
market['middleman'] = fuzz.trimf(market.universe, [1, 1, 2])
market['processor'] = fuzz.trimf(market.universe, [2, 2, 2])

land_acreage['small'] = fuzz.trimf(land_acreage.universe, [0, 1, 2])
land_acreage['medium'] = fuzz.trimf(land_acreage.universe, [1, 3, 4])
land_acreage['large'] = fuzz.trimf(land_acreage.universe, [4, 6, 6])

rainfall['low'] = fuzz.trimf(rainfall.universe, [0, 0, 200])
rainfall['medium'] = fuzz.trimf(rainfall.universe, [150, 300, 450])
rainfall['high'] = fuzz.trimf(rainfall.universe, [400, 500, 500])

temperature['low'] = fuzz.trimf(temperature.universe, [0, 0, 20])
temperature['medium'] = fuzz.trimf(temperature.universe, [15, 25, 35])
temperature['high'] = fuzz.trimf(temperature.universe, [30, 50, 50])

income_source['unstable'] = fuzz.trimf(income_source.universe, [0, 0, 50])
income_source['stable'] = fuzz.trimf(income_source.universe, [30, 50, 100])

annual_expenditure['low'] = fuzz.trimf(annual_expenditure.universe, [0, 0, 300000])
annual_expenditure['medium'] = fuzz.trimf(annual_expenditure.universe, [250000, 500000, 700000])
annual_expenditure['high'] = fuzz.trimf(annual_expenditure.universe, [600000, 1000000, 1000000])

insurance['no'] = fuzz.trimf(insurance.universe, [0, 0, 1])
insurance['yes'] = fuzz.trimf(insurance.universe, [1, 1, 1])

# Define membership functions for risk output
risk['low'] = fuzz.trimf(risk.universe, [0, 0, 4])
risk['medium'] = fuzz.trimf(risk.universe, [3, 5, 7])
risk['high'] = fuzz.trimf(risk.universe, [6, 10, 10])

# Define fuzzy rules
rules = [
    ctrl.Rule(age['young'] & crop_season['high'] & guarantor['yes'] & financial_records['good'] & farming_history['good'] & bank_statements['good'] & market['processor'], risk['low']),
    ctrl.Rule(age['old'] & crop_season['low'] & guarantor['no'] & financial_records['poor'], risk['high']),
    ctrl.Rule(age['middle_aged'] & crop_season['medium'] & financial_records['average'] & market['middleman'] & rainfall['medium'], risk['medium']),
    # Add more rules as needed to cover different combinations
]

# Create control system
risk_ctrl = ctrl.ControlSystem(rules)
risk_simulation = ctrl.ControlSystemSimulation(risk_ctrl)

# Function to assess risk
def assess_risk(farmer_data):
    for param, value in farmer_data.items():
        risk_simulation.input[param] = value
    risk_simulation.compute()
    return risk_simulation.output['risk']

# Example farmer data (you can create real inputs as per your use case)
farmer_data = {
    'age': 30,
    'crop_season': 70,
    'guarantor': 1,
    'financial_records': 80,
    'farming_history': 90,
    'bank_statements': 85,
    'inputs_required': 40,
    'market': 2,
    'land_acreage': 25,
    'rainfall': 300,
    'temperature': 25,
    'income_source': 70,
    'annual_expenditure': 400000,
    'insurance': 1
}

# Calculate and print risk
risk_score = assess_risk(farmer_data)
print(f"Farmer's risk score is {risk_score:.2f}/10")
