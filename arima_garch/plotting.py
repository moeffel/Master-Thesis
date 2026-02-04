# Plotting utilities
import logging
from typing import Any, Optional, List, Dict
import os
import math
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.table import Table
    PLOT_AVAILABLE = True
except Exception:
    matplotlib = None
    plt = None
    Table = None
    PLOT_AVAILABLE = False
try:
    import dataframe_image as dfi
    DATAFRAME_IMAGE_AVAILABLE = True
except Exception:
    dfi = None
    DATAFRAME_IMAGE_AVAILABLE = False
import numpy as np
try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
except Exception:
    plot_acf = None
    plot_pacf = None
log = logging.getLogger(__name__)

def get_col_widths(data:List[List[Any]],sf:float=0.018,min_w:float=0.08,max_w:float=0.3)->List[float]:
    """Calculate appropriate column widths for matplotlib table based on content."""
    if not data or not data[0]: return []
    num_cols=len(data[0])
    max_len=[0]*num_cols
    for row in data:
        for j in range(num_cols): max_len[j]=max(max_len[j], math.ceil(len(str(row[j]).strip())))
    # Scale length by factor, apply min/max constraints
    return [min(max_w,max(min_w,l*sf)) for l in max_len]

def create_parameter_table_png(arima:Optional[Any], garch:Optional[Any], filename:str, title:str)->bool:
    """Creates a PNG image containing formatted parameter tables from ARIMA and GARCH results."""
    if not PLOT_AVAILABLE or plt is None or Table is None: log.warning("Plotting disabled, cannot create parameter table PNG."); return False
    if arima is None and garch is None: log.warning("No model results provided for parameter table."); return False

    tables=[]
    summaries=[]
    try: # Extract ARIMA parameters if available
        if arima and hasattr(arima,'summary'):
            s=arima.summary()
            if hasattr(s,'tables') and len(s.tables)>1: # Table 1 usually contains coefficients
                d=[[str(c).strip() for c in r] for r in s.tables[1].data] # Extract data, convert to string, strip whitespace
                if len(d)>1: # Ensure there's data beyond the header
                    tables.append({"title":"ARIMA Parameters","data":d})
                # Add summary stats
                summaries.append(f"ARIMA LLF:{getattr(arima,'llf',np.nan):.2f} AIC:{getattr(arima,'aic',np.nan):.2f} BIC:{getattr(arima,'bic',np.nan):.2f}")
    except Exception as e: log.error(f"Error extracting ARIMA table data: {e}")

    try: # Extract GARCH parameters if available
        if garch and hasattr(garch,'summary'):
            s=garch.summary()
            if hasattr(s,'tables') and len(s.tables)>1: # Table 1 for vol params
                gd=[[str(c).strip() for c in r] for r in s.tables[1].data]
                if len(s.tables)>2: # Table 2 often has distribution params
                    try:
                        gd.extend([[str(c).strip() for c in r] for r in s.tables[2].data])
                    except IndexError: log.debug("No GARCH distribution parameter table found in summary.")
                    except Exception as egd: log.warning(f"Error getting GARCH distribution params: {egd}")
                if len(gd)>1:
                    tables.append({"title":"GARCH Parameters (Vol+Dist)","data":gd})
                summaries.append(f"GARCH LLF:{getattr(garch,'loglikelihood',np.nan):.2f} AIC:{getattr(garch,'aic',np.nan):.2f} BIC:{getattr(garch,'bic',np.nan):.2f}")
    except Exception as e: log.error(f"Error extracting GARCH table data: {e}")

    if not tables: log.warning("No valid parameter tables extracted from model summaries."); return False

    num_rows_total = sum(len(t['data']) for t in tables) # Total rows across all tables
    fig_height = 1.5 + len(tables)*0.8 + num_rows_total*0.3 + len(summaries)*0.2 # Estimate figure height
    fig,ax=plt.subplots(figsize=(10,fig_height))
    ax.axis('off') # Hide axes

    current_y=0.98
    fig.text(0.5,current_y,title,ha='center',va='top',fontsize=14,weight='bold')
    current_y-=0.10

    # Add summary stats text
    if summaries:
        summary_text="\n".join(summaries)
        fig.text(0.05,current_y,summary_text,ha='left',va='top',fontsize=9,family='monospace')
        current_y-=(len(summaries)*0.02 + 0.05)

    # Add each table
    for table_info in tables:
        table_data=table_info['data']
        table_title=table_info['title']
        if not table_data or len(table_data)<=1: continue # Skip empty tables

        fig.text(0.5,current_y,table_title,ha='center',va='top',fontsize=11,weight='bold')
        current_y-=0.06

        rows,cols=len(table_data),len(table_data[0])
        try:
            col_widths=get_col_widths(table_data)
            # Estimate table height and width, position it
            table_height=min(0.9/max(1,len(tables)),rows*0.04) # Allocate vertical space
            bbox_width=min(0.95,sum(col_widths)+0.1) # Calculate required width
            bbox_x=max(0.02,0.5-bbox_width/2) # Center horizontally
            bbox=[bbox_x,max(0.01,current_y-table_height),bbox_width,table_height]

            tab=ax.table(cellText=table_data[1:],colLabels=table_data[0],colWidths=col_widths,
                           cellLoc='center',loc='upper center',bbox=bbox)
            tab.auto_set_font_size(False)
            tab.set_fontsize(7)

            # Style header
            for j in range(cols):
                cell=tab.get_celld().get((0,j)) # Header cell is row 0
                if cell: cell.set_text_props(weight='bold',color='white'); cell.set_facecolor('#000080') # Dark blue header

            current_y-=(table_height + 0.07) # Move down for next element
        except Exception as te: log.error(f"Error creating table '{table_title}' in PNG: {te}"); ax.text(0.5,current_y,f"Error rendering: {table_title}",ha='center',va='top',color='red'); current_y-=0.1

    try:
        fig.subplots_adjust(top=0.92,bottom=0.02,left=0.02,right=0.98) # Adjust layout
        fig.savefig(filename,dpi=150,bbox_inches='tight')
        log.info(f"Parameter table PNG saved: {filename}")
        plt.close(fig)
        return True
    except Exception as e: log.error(f"Error saving PNG {filename}: {e}"); plt.close(fig); return False

def print_model_summary_console(arima:Optional[Any], garch:Optional[Any], coin_id:str)->None:
    """Prints a formatted summary of ARIMA and GARCH model parameters to the console."""
    print(f"\n--- [{coin_id.upper()}] Model Parameter Summary ---")

    def _print(title:str, result_obj:Any):
        print(f"\n{title}:")
        if result_obj is None: print("  Model N/A."); return

        try:
            summary=result_obj.summary()
            # Check if summary has the expected table structure
            if not hasattr(summary,'tables') or len(summary.tables)<2:
                # Fallback: print basic info if summary structure is unexpected
                print(f"  Incomplete Summary. Basic Info:")
                print(f"   Class: {type(result_obj)}")
                llf = getattr(result_obj,'llf',getattr(result_obj,'loglikelihood',np.nan))
                aic = getattr(result_obj,'aic',np.nan)
                bic = getattr(result_obj,'bic',np.nan)
                nobs = getattr(result_obj,'nobs',np.nan)
                params = getattr(result_obj,'params', None)
                pvals = getattr(result_obj,'pvalues', None)
                print(f"   LLF: {llf:.3f}, AIC: {aic:.3f}, BIC: {bic:.3f}, Nobs: {nobs}")
                if params is not None:
                     print("   Parameters:")
                     params_df = pd.DataFrame({'coef': params})
                     if pvals is not None and len(pvals) == len(params):
                         params_df['P>|z|'] = pvals
                         # Format p-value nicely
                         params_df['P>|z|'] = pd.to_numeric(params_df['P>|z|'], errors='coerce').map('{:.3f}'.format)
                     print(params_df.to_string(float_format="{:.4f}".format, na_rep='N/A'))
                return

            param_table=summary.tables[1] # Usually parameters
            try:
                # Try parsing the HTML representation for better formatting
                html_content = param_table.as_html()
                df = None
                if not html_content or "</table>" not in html_content:
                     log.warning(f"Summary Table 1 ({title}) empty/invalid. Raw:")
                     print(param_table.as_text())
                else:
                    try:
                        # Requires lxml or html5lib
                        df = pd.read_html(html_content, header=0, index_col=0)[0]
                    except ImportError:
                        log.error("Pandas read_html requires 'lxml' or 'html5lib'. Install one.")
                        print("  ERROR: Could not read HTML table (missing library).")
                        print("  Raw:\n"+param_table.as_text())
                    except ValueError as ve_html:
                        log.error(f"Pandas read_html ValueError: {ve_html}")
                        print("  ERROR: Could not read HTML table (Pandas error).")
                        print("  Raw:\n"+param_table.as_text())

                if df is None:
                    print("  Could not parse Summary Table 1.")
                else:
                     # Select and format common columns
                     cols=['coef','std err','z','P>|z|']
                     existing_cols=[c for c in cols if c in df.columns]
                     if not existing_cols:
                         print(f"  Standard columns not found. Raw:\n{df.to_string(na_rep='N/A')}")
                     else:
                         if 'P>|z|' in df.columns:
                             # Add significance stars
                             df['p_num']=pd.to_numeric(df['P>|z|'],errors='coerce')
                             df['Sig.'] = df['p_num'].apply(lambda p: "***" if pd.notna(p) and p<0.01 else ("** " if pd.notna(p) and p<0.05 else ("*  " if pd.notna(p) and p<0.10 else "   ")))
                             df['P>|z|']=df['p_num'].map('{:.3f}'.format) # Format p-value
                             existing_cols.append('Sig.')
                             df=df.drop(columns=['p_num'])
                         # Print selected columns
                         print_cols = [c for c in existing_cols if c in df.columns]
                         print(df[print_cols].to_string(float_format="{:.4f}".format,na_rep='N/A'))

            except Exception as e_parse:
                 print(f"  Error parsing Table 1: {e_parse}. Raw:\n{param_table.as_text()}")

            # Print GARCH distribution parameters if available (usually Table 2)
            if title.startswith("GARCH") and len(summary.tables)>2:
                print("\n  Distribution Parameters:")
                dist_table=summary.tables[2]
                try:
                    html_dt = dist_table.as_html()
                    dist_df = None
                    if not html_dt or "</table>" not in html_dt:
                        log.warning(f"Distribution Params Table empty. Raw:")
                        print(dist_table.as_text())
                    else:
                        try:
                           dist_df = pd.read_html(html_dt,header=0,index_col=0)[0]
                        except ImportError:
                           log.error("Pandas read_html requires 'lxml' or 'html5lib'. Install one.")
                           print("  ERROR: Could not read HTML table (missing library).")
                           print("  Raw:\n"+dist_table.as_text())
                        except ValueError as ve_html_dist:
                           log.error(f"Pandas read_html ValueError (Dist): {ve_html_dist}")
                           print("  ERROR: Could not read HTML table (Pandas error).")
                           print("  Raw:\n"+dist_table.as_text())

                        if dist_df is not None and not dist_df.empty:
                           print(dist_df.to_string(float_format="{:.4f}".format,na_rep='N/A'))
                        elif dist_df is None:
                           pass # Error already printed
                        else: # Parsed but empty
                            print("  Could not parse Distribution Params table (empty).")

                except Exception as e_parse_dist:
                    print(f"  Error parsing Distribution Params: {e_parse_dist}. Raw:\n{dist_table.as_text()}")

            # Print overall model stats
            llf=getattr(result_obj,'llf',getattr(result_obj,'loglikelihood',np.nan))
            aic=getattr(result_obj,'aic',np.nan)
            bic=getattr(result_obj,'bic',np.nan)
            print(f"\n  LLF: {llf:.3f}, AIC: {aic:.3f}, BIC: {bic:.3f}")

        except Exception as e:
             print(f"  Unexpected error generating summary: {e}")
             print(f"  Object representation: {result_obj}")

    _print("ARIMA Parameters",arima)
    _print("GARCH Parameters",garch)
    print("--- End Parameter Summary ---")

def plot_parameter_stability(param_data:Dict[str,list], dates:list, filename:str, title:str)->bool:
    """Plots the evolution of selected model parameters over time (for backtesting)."""
    if not PLOT_AVAILABLE or plt is None: log.warning("Plotting disabled, cannot create stability plot."); return False
    if not param_data or not dates: log.warning("No data provided for stability plot."); return False

    try: plot_dates=pd.to_datetime(dates)
    except Exception as e: log.error(f"Date conversion failed for stability plot: {e}"); return False

    # Filter for parameters that have data and are not all NaN
    valid_params={k:v for k,v in param_data.items() if len(v)==len(plot_dates) and any(pd.notna(x) for x in v)}
    if not valid_params: log.warning("No valid parameters found for stability plot."); return False

    n_params=len(valid_params)
    fig_height = max(5.0, n_params * 2.0) # Adjust height based on number of parameters
    fig,axes=plt.subplots(n_params,1,figsize=(10, fig_height),sharex=True,squeeze=False)
    axes=axes.flatten() # Ensure axes is always a 1D array

    fig.suptitle(title,fontsize=14,y=0.99)
    param_names=list(valid_params.keys())

    for i,name in enumerate(param_names):
        ax=axes[i]
        values=valid_params[name]
        # Convert to numeric, create Series with dates, drop NaNs for plotting
        param_series=pd.Series(pd.to_numeric(values,errors='coerce'),index=plot_dates).dropna()

        if not param_series.empty:
            ax.plot(param_series.index,param_series.values,marker='.',linestyle='-',markersize=3,label=name,lw=1.2,color='black')
            mean_val=param_series.mean()
            std_dev = param_series.std()
            ax.axhline(mean_val,color='r',linestyle='--',linewidth=0.8,label=f'Mean: {mean_val:.4f}')
            # Add +/- 1 std dev shading if std dev is meaningful
            if pd.notna(std_dev) and std_dev > 1e-8:
                ax.fill_between(param_series.index, mean_val - std_dev, mean_val + std_dev, color='red', alpha=0.1, label=f'+/- 1 StdDev ({std_dev:.4f})')
            ax.legend(loc='best',fontsize='small')
            ax.grid(True,linestyle=':',alpha=0.6)
        else:
            # Handle case where parameter is all NaN after conversion
            ax.text(0.5,0.5,f'No valid data for {name}',ha='center',va='center',transform=ax.transAxes, color='grey')
            ax.grid(True,linestyle=':',alpha=0.6)

        # Clean up parameter name for display
        display_name = name.replace('[','').replace(']','') # Remove brackets often used in statsmodels names
        ax.set_ylabel(display_name)

    # Configure bottom axis
    axes[-1].set_xlabel("Date")
    plt.setp(axes[-1].xaxis.get_majorticklabels(),rotation=45,ha='right')
    fig.tight_layout(pad=0.5)
    try:
        fig.savefig(filename, dpi=100)
        log.info(f"Stability plot saved: {filename}")
        plt.close(fig)
        return True
    except Exception as e:
        log.error(f"Error saving stability plot {filename}: {e}")
        plt.close(fig)
        return False

# --- Data Handling ---

def plot_combined_qq(all_plot_data: list, plot_dir_base: str):
    """Generates a single 2x2 figure with Q-Q plots for all four assets."""
    if not PLOT_AVAILABLE or plt is None or sm is None or len(all_plot_data) != 4:
        log.warning("Skipping combined Q-Q plot generation (requirements not met).")
        return

    log.info("Generating combined Q-Q plot for all assets...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten() # Macht die Indizierung einfacher

    for i, data in enumerate(all_plot_data):
        ax = axes[i]
        qq_series = data['qq_series']
        coin_id = data['coin_id']

        if not qq_series.empty:
            sm.qqplot(qq_series, line='s', fit=True, ax=ax)
            ax.set_title(f"{coin_id.upper()}", fontsize=14)
            ax.get_lines()[0].set_markerfacecolor('black') # Datenpunkte
            ax.get_lines()[0].set_markeredgecolor('black')
            ax.get_lines()[0].set_markersize(4.0)
            ax.get_lines()[1].set_color('red') # Referenzlinie
            ax.get_lines()[1].set_linewidth(2.0)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax.set_title(f"{coin_id.upper()}", fontsize=14)

    fig.suptitle('Q-Q Plots of Log Returns vs. Normal Distribution', fontsize=18, y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.93]) # Platz für den suptitle lassen

    try:
        filename = os.path.join(plot_dir_base, "DIAG_combined_qq_plots.png")
        fig.savefig(filename, dpi=200)
        log.info(f"Combined Q-Q plot saved: {filename}")
        plt.close(fig)
    except Exception as e:
        log.error(f"Failed to save combined Q-Q plot: {e}")
        plt.close(fig)

def plot_combined_acf_pacf(all_plot_data: list, plot_dir_base: str):
    """Generates a single 4x2 figure with ACF and PACF plots for all four assets."""
    if not PLOT_AVAILABLE or plt is None or len(all_plot_data) != 4:
        log.warning("Skipping combined ACF/PACF plot generation (requirements not met).")
        return

    log.info("Generating combined ACF/PACF plot for all assets...")
    fig, axes = plt.subplots(4, 2, figsize=(14, 16)) # 4 Zeilen, 2 Spalten

    for i, data in enumerate(all_plot_data):
        series = data['acf_pacf_series']
        coin_id = data['coin_id']
        d_order = data['d_final']
        
        # ACF Plot
        ax_acf = axes[i, 0]
        if not series.empty:
            plot_acf(series, ax=ax_acf, lags=40, zero=False)
        ax_acf.set_title(f"{coin_id.upper()} - ACF (d={d_order})")
        ax_acf.set_ylim(-1.0, 1.0)

        # PACF Plot
        ax_pacf = axes[i, 1]
        if not series.empty:
            plot_pacf(series, ax=ax_pacf, lags=40, zero=False, method='ywm')
        ax_pacf.set_title(f"{coin_id.upper()} - PACF (d={d_order})")
        ax_pacf.set_ylim(-1.0, 1.0)

    fig.suptitle('Autocorrelation and Partial Autocorrelation of Log Returns', fontsize=18, y=0.96)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    try:
        filename = os.path.join(plot_dir_base, "DIAG_combined_acf_pacf_plots.png")
        fig.savefig(filename, dpi=200)
        log.info(f"Combined ACF/PACF plot saved: {filename}")
        plt.close(fig)
    except Exception as e:
        log.error(f"Failed to save combined ACF/PACF plot: {e}")
        plt.close(fig)
        
# ==============================================================================
# --- MAIN ANALYSIS FUNCTION FOR ONE COIN ---
# ==============================================================================
