import streamlit as st
import numpy as np
from pyXSteam.XSteam import XSteam
from fluprodia import FluidPropertyDiagram
import matplotlib.pyplot as plt

# Function to calculate the Rankine cycle parameters
def calculate_rankine_cycle(p1, p2, T3, HPturbeff, LPturbeff):
    """Calculate all parameters of the Rankine cycle."""
    steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)
    
    try:
        # Ensure inputs are floats to avoid type errors
        p1 = float(p1)
        p2 = float(p2)
        T3 = float(T3)
        HPturbeff = float(HPturbeff)
        LPturbeff = float(LPturbeff)

        # Point 1: Condenser outlet (saturated liquid)
        s1 = steamTable.sL_p(p1)  # Specific entropy at liquid state
        h1 = steamTable.hL_p(p1)  # Specific enthalpy at liquid state
        T1 = steamTable.tsat_p(p1)  # Saturation temperature at p1

        # Point 2: Pump outlet (compressed liquid)
        s2 = s1  # Isentropic compression (constant entropy)
        v1 = 1 / steamTable.rhoL_p(p1)  # Specific volume of liquid at p1
        w_p = v1 * (p2 - p1) * 100  # Work done by the pump in kJ/kg
        h2 = h1 + w_p  # Enthalpy after pump work
        T2 = steamTable.t_ph(p2, h2)  # Temperature at pressure p2 and enthalpy h2

        if np.isnan(T2) or np.isnan(h2):
            raise ValueError("Invalid pump outlet enthalpy or temperature.")

        # Point 2': Boiler inlet (saturated liquid)
        h2dash = steamTable.hL_p(p2)  # Enthalpy at saturated liquid state at p2
        s2dash = steamTable.sL_p(p2)  # Entropy at saturated liquid state at p2
        T2dash = steamTable.tsat_p(p2)  # Saturation temperature at p2
        
        h3dash = steamTable.hV_p(p2)
        s3dash = steamTable.sV_p(p2)
        T3dash = T2dash
        
        # Point 3: Boiler outlet (superheated steam)
        h3 = steamTable.h_pt(p2, T3)  # Enthalpy at pressure p2 and temperature T3
        s3 = steamTable.s_pt(p2, T3)  # Entropy at pressure p2 and temperature T3

        if np.isnan(h3) or np.isnan(s3):
            raise ValueError("Invalid boiler outlet enthalpy or temperature.")

        # Point 4: HP Turbine outlet
        p4 = p2 / 8  # Pressure drop after HP turbine
        h4s = steamTable.h_ps(p4, s3)  # Isentropic enthalpy at p4
        h4 = h3 - HPturbeff * (h3 - h4s)  # Real enthalpy after HP turbine
        s4 = steamTable.s_ph(p4, h4)  # Entropy at p4 and h4
        T4 = steamTable.t_ph(p4, h4)  # Temperature at p4 and h4

        if np.isnan(h4) or np.isnan(s4):
            raise ValueError("Invalid HP turbine outlet enthalpy or temperature.")

        # Work done by HP turbine
        w_HPt = h3 - h4  # Work output of the HP turbine in kJ/kg

        # Point 5: Reheat (after reheating in boiler)
        T5 = T3  # Assume reheating brings temperature back to T3
        h5 = steamTable.h_pt(p4, T5)  # Enthalpy at p4 and reheated temperature T5
        s5 = steamTable.s_pt(p4, T5)  # Entropy at p4 and reheated temperature T5

        if np.isnan(h5) or np.isnan(s5):
            raise ValueError("Invalid reheat enthalpy or temperature.")

        # Point 6: LP Turbine outlet
        p6 = p1  # Condenser pressure
        h6s = steamTable.h_ps(p6, s5)  # Isentropic enthalpy at p6
        h6 = h5 - LPturbeff * (h5 - h6s)  # Real enthalpy after LP turbine
        s6 = steamTable.s_ph(p6, h6)  # Entropy at p6 and h6
        T6 = steamTable.t_ph(p6, h6)  # Temperature at p6 and h6
        x6 = steamTable.x_ph(p6, h6)  # Quality of steam at point 6

        if np.isnan(h6) or np.isnan(s6) or np.isnan(x6):
            raise ValueError("Invalid LP turbine outlet enthalpy, temperature, or steam quality.")

        # Work done by LP turbine
        w_LPt = h5 - h6  # Work output of the LP turbine in kJ/kg

        # Heat transfer in the boiler and reheater
        q_H = (h3 - h2) + (h5 - h4)  # Heat input in the boiler and reheater
        q_L = h6 - h1  # Heat rejected in the condenser

        # Thermal efficiency calculation
        eta_th = ((w_HPt + w_LPt) - w_p) / q_H * 100  # Efficiency in %

        # Heat rate in kJ/kWh
        HRcycle = 3600 / (eta_th / 100)
    
    except Exception as e:
        raise ValueError(f"Error during calculation: {str(e)}")

    return {
        'points': {
            '1': {'T': T1, 'p': p1, 'h': h1, 's': s1},
            '2': {'T': T2, 'p': p2, 'h': h2, 's': s2},
            '2dash': {'T': T2dash, 'p': p2, 'h': h2dash, 's': s2dash},
            '3dash': {'T': T3dash, 'p': p2, 'h': h3dash, 's': s3dash},
            '3': {'T': T3, 'p': p2, 'h': h3, 's': s3},
            '4': {'T': T4, 'p': p4, 'h': h4, 's': s4},
            '5': {'T': T5, 'p': p4, 'h': h5, 's': s5},
            '6': {'T': T6, 'p': p6, 'h': h6, 's': s6, 'x': x6}
        },
        'work': {
            'pump': w_p,
            'hp_turbine': w_HPt,
            'lp_turbine': w_LPt,
            'total_turbine': w_HPt + w_LPt
        },
        'heat': {
            'input': q_H,
            'rejected': q_L
        },
        'efficiency': {
            'thermal': eta_th,
            'heat_rate': HRcycle
        }
    }


def plot_ts_diagram(results):
    """Create T-s diagram using FluidPropertyDiagram"""
    
    try:
        # Initialize the fluid property diagram for water
        diagram = FluidPropertyDiagram(fluid='H2O')
        diagram.set_unit_system(T='°C', s='kJ/kgK', p='bar')
        
        # Set the isolines for temperature and specific entropy
        iso_T = np.arange(100, 700, 50)  # Temperature isolines from 100 to 700°C
        iso_s = np.arange(0.5, 8, 0.5)  # Entropy isolines from 0.5 to 8 kJ/kg·K
        
        # Set the isolines in the diagram
        diagram.set_isolines(T=iso_T, s=iso_s)
        
        # Calculate the isolines
        diagram.calc_isolines()

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Set the limits for the diagram
        x_min, x_max = 0.5, 8  # Specific entropy limits in kJ/kg·K
        y_min, y_max = 0, 700  # Temperature limits in °C
        
        # Draw the T-s diagram with isolines and the set limits
        diagram.draw_isolines(diagram_type='Ts', fig=fig, ax=ax, 
                              x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

        # Extract the Rankine cycle points
        points = results['points']
        
        # Ensure all necessary points have valid values
        required_points = ['1', '2dash','3dash', '3', '4', '5', '6']
        s_values = []
        T_values = []
        
        for point in required_points:
            s = points[point].get('s')
            T = points[point].get('T')
            if s is not None and T is not None:
                s_values.append(s)
                T_values.append(T)
            else:
                raise ValueError(f"Missing value at point {point}: s={s}, T={T}")
        
        # Close the cycle by connecting back to the first point
        s_values.append(s_values[0])
        T_values.append(T_values[0])

        # Plot the cycle points and connect them with red lines
        ax.plot(s_values, T_values, 'r-', label='Rankine Cycle')
        
        # Annotate the points on the cycle
        for i, (s, T) in enumerate(zip(s_values[:-1], T_values[:-1])):
            point_name = ['1', '2\'','3\'' ,'3', '4', '5', '6'][i]
            ax.annotate(f'Point {point_name}', 
                        xy=(s, T), 
                        xytext=(10, 10), 
                        textcoords='offset points', 
                        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
        
        # Set labels and title for the plot
        ax.set_xlabel('Specific Entropy [kJ/kg·K]')
        ax.set_ylabel('Temperature [°C]')
        ax.set_title('T-s Diagram of the Rankine Cycle')
        ax.grid(True)
        ax.legend(loc='upper left')
        
        st.pyplot(fig)  # This is the missing part

        # Return the figure for rendering in Streamlit
        return fig
    
    except Exception as e:
        st.error(f"An error occurred in the T-s diagram plotting: {str(e)}")




# Main function to run the Streamlit app
def main():
    st.title('Rankine Reheat Cycle Analysis')
    
    st.sidebar.header('Input Parameters')
    
    # Input parameters with more intuitive ranges and step sizes
    p1 = st.sidebar.number_input('Condenser Pressure (p1) [bar]', 
                                min_value=0.01, max_value=1.0, 
                                value=0.06, step=0.01, 
                                format="%.3f")
    
    p2 = st.sidebar.number_input('Boiler Pressure (p2) [bar]', 
                                min_value=50.0, max_value=300.0, 
                                value=150.0, step=10.0)
    
    T3 = st.sidebar.number_input('Maximum Temperature (T3) [°C]', 
                                min_value=300.0, max_value=700.0, 
                                value=540.0, step=10.0)
    
    HPturbeff = st.sidebar.slider('HP Turbine Efficiency', 
                                 min_value=0.70, max_value=1.0, 
                                 value=0.90, step=0.01)
    
    LPturbeff = st.sidebar.slider('LP Turbine Efficiency', 
                                 min_value=0.70, max_value=1.0, 
                                 value=0.90, step=0.01)
    
    try:
        # Calculate results
        results = calculate_rankine_cycle(p1, p2, T3, HPturbeff, LPturbeff)
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(["Performance Metrics", "T-s Diagram", "State Points"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader('Work and Power')
                st.metric("Pump Work", f"{results['work']['pump']:.1f} kJ/kg")
                st.metric("HP Turbine Work", f"{results['work']['hp_turbine']:.1f} kJ/kg")
                st.metric("LP Turbine Work", f"{results['work']['lp_turbine']:.1f} kJ/kg")
                st.metric("Total Turbine Work", f"{results['work']['total_turbine']:.1f} kJ/kg")
            
            with col2:
                st.subheader('Heat and Efficiency')
                st.metric("Heat Input", f"{results['heat']['input']:.1f} kJ/kg")
                st.metric("Heat Rejected", f"{results['heat']['rejected']:.1f} kJ/kg")
                st.metric("Thermal Efficiency", f"{results['efficiency']['thermal']:.1f}%")
                st.metric("Heat Rate", f"{results['efficiency']['heat_rate']:.1f} kJ/kWh")
        
        with tab2:
            st.pyplot(plot_ts_diagram(results))
        
        with tab3:
            for point, values in results['points'].items():
                with st.expander(f'Point {point}'):
                    cols = st.columns(4)
                    for i, (param, value) in enumerate(values.items()):
                        cols[i % 4].metric(
                            label=param.upper(),
                            value=f"{value:.3f}"
                        )
    
    except Exception as e:
        st.error(f"An error occurred in the calculations: {str(e)}")

if __name__ == '__main__':
    main()
