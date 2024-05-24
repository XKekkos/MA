import io, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive, interact
import ipywidgets as widgets

df = pd.read_csv(io.StringIO('''
TICKET REVENUE PER CAPITA,
Avg Ticket Price Net Of Tax Per Paying Patron,$22.12
Facility Charge,$2.91
Service Charge,$1.96
Total Ticket Sales Revenue,$26.99
ANCILLIARY REVENUE PER CAPITA,
Parking,$1.91
Food Concession,$7.66
Merchandise,$3.52
Total Ancillary Revenue,$13.09
FIXED COSTS,
Parking (Fixed),"$2,471.30 "
Food Concession (Fixed),"$35,428.70 "
Merchandise (Fixed),"$14,183.20 "
Production,"$15,506.00 "
Operations,"$14,991.00 "
Advertising,"$20,030.00 "
Artists,"$160,635.00 "
Total Fixed Costs,"$263,245.00 "
VARIABLE COSTS PER CAPITA,
Parking (Variable),$0.19
Food Concession (Variable),$0.77
Merchandise (Variable),$0.35
Other Variable Costs,$1.74
Total Variable Costs,$3.05
"2a,2b result"
Break-even point,6658
Target Profit $30000 with 40% tax,7923
'''), index_col=0, header=None)

df = df[1].str.strip('$').str.replace(',', '').astype(float)

options = ['Avg Ticket Price Net Of Tax Per Paying Patron', 'Parking (Fixed)',
           'Food Concession (Fixed)', 'Merchandise (Fixed)', 'Production',
           'Operations', 'Advertising', 'Artists', 'Parking (Variable)',
           'Food Concession (Variable)', 'Merchandise (Variable)', 'Other Variable Costs',]

def new_ticket_quantity(var: str=None, p: float=0.0, PT: bool=False) -> float:
    # Make a copy of the Series
    df_new = df.copy()

    if var:
        # Define new value
        df_new[var] = df[var]*(1+p)

    if PT:
        PT = 50000

    # Calculate Total Revenue = Paid Revenue + Comp Revenue
    TR = sum(df_new.iloc[1:4].values) + 1.25*sum(df_new.iloc[6:9].values)

    # Calculate Variable Costs
    VC = 1.25*sum(df_new.iloc[20:24].values)

    # Calculate Total Fixed Costs
    FC = sum(df_new.iloc[11:18].values) + PT

    # Caclulate q
    q = FC / (TR - VC)

    # Return rounded up version of q
    return math.ceil(q)

def percentage_change(p: float=0.0, var: str=None, PT: int=0) -> float:
    # Get new quantity
    new = new_ticket_quantity(p=p, var=var, PT=PT)

    # Get old quantity
    old = new_ticket_quantity(PT=PT)

    # Return percentage change
    return round(100*(new-old)/old,2)


def plot_differences():
    %matplotlib inline
    @interact(var=widgets.Dropdown(options=options,
                                value=options[0],
                                description='Variable'),
            PT=widgets.Checkbox(value=False,
                                description='Profit Target: $30,000',
                                indent=False),
            y=widgets.ToggleButtons(options=['Change', '% Change'],
                                    description='y-axis',
                                    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    ))
    def plot(var: str=None, PT: bool=False, y: str='Change') -> None:
        # Get possible price changes
        ps = np.linspace(-1, 1, 101)

        # Get y values
        initial_value = new_ticket_quantity(PT=PT)
        if y == 'Change':
            change = [new_ticket_quantity(var=var, p=p, PT=PT) - initial_value for p in ps]

        if y == '% Change':
            change = [percentage_change(var=var, p=p, PT=PT) for p in ps]

        plt.style.use('ggplot')

        plt.figure(figsize=(10,6))
        plt.plot(100*ps, change)

        # Change labels
        plt.xlabel(f'Percentage Change in {var}')
        plt.ylabel(f'{y} in number of tickets')

        plt.show()
