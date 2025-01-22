import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlos_bench.storage import from_config
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

# ------------------------------------------------------------------------------
# Configuration and Setup
# ------------------------------------------------------------------------------
@dataclass
class Config:
    """Application configuration settings"""
    TITLE: str = "MySQL Configuration Analysis Dashboard"
    DESCRIPTION: str = "Analyze and optimize MySQL database configurations"
    DB_CONFIG_PATH: str = "storage/sqlite.jsonc"
    THEME: str = "plotly_white"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize page configuration first
st.set_page_config(page_title=Config.TITLE, layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------------------
# Data Loading and Processing
# ------------------------------------------------------------------------------
@st.cache_resource
def load_data():
    """Load and cache the database storage"""
    try:
        storage = from_config(config=Config.DB_CONFIG_PATH)
        logger.info("Successfully loaded database storage")
        return storage
    except Exception as e:
        logger.error(f"Failed to load storage: {e}")
        st.error(f"Failed to load storage: {str(e)}")
        return None

class DataProcessor:
    """Handle data processing operations"""

    @staticmethod
    def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Extract configuration and result columns"""
        config_cols = [col for col in df.columns if col.startswith("config.")]
        result_cols = [col for col in df.columns if col.startswith("result.")]
        return config_cols, result_cols

    @staticmethod
    def calculate_stats(df: pd.DataFrame) -> Dict:
        """Calculate key statistics from the dataset"""
        total = len(df)
        success = df["status"].value_counts().get("SUCCESS", 0)
        failed = df["status"].value_counts().get("FAILED", 0)
        success_rate = (success / total * 100) if total > 0 else 0

        return {
            "total": total,
            "success": success,
            "failed": failed,
            "success_rate": success_rate,
            "failure_rate": 100 - success_rate
        }

    @staticmethod
    def perform_clustering(df: pd.DataFrame, columns: List[str], n_clusters: int) -> Dict:
        """Perform KMeans clustering on selected columns"""
        X = df[columns].fillna(df[columns].mean())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        return {
            "labels": clusters,
            "centroids": kmeans.cluster_centers_,
            "inertia": kmeans.inertia_
        }

class Visualizer:
    """Handle data visualization"""

    @staticmethod
    def plot_trial_outcomes(stats: Dict) -> go.Figure:
        """Create pie chart of trial outcomes"""
        return px.pie(
            names=["Success", "Failed"],
            values=[stats["success"], stats["failed"]],
            title="Trial Outcomes",
            color_discrete_map={"Success": "green", "Failed": "red"},
            template=Config.THEME
        )

    @staticmethod
    def plot_metric_distribution(df: pd.DataFrame, metric: str) -> go.Figure:
        """Create distribution plot for a metric"""
        return px.histogram(
            df, x=metric,
            title=f"Distribution of {metric.replace('result.', '').replace('_', ' ').title()}",
            template=Config.THEME,
            marginal="box"
        )

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame, columns: List[str]) -> go.Figure:
        """Create correlation heatmap"""
        corr_matrix = df[columns].corr()
        return px.imshow(
            corr_matrix,
            title="Correlation Heatmap",
            template=Config.THEME,
            aspect="auto"
        )

class Dashboard:
    """Main dashboard application"""

    def __init__(self):
        self.storage = load_data()
        self.processor = DataProcessor()
        self.visualizer = Visualizer()

    def run(self):
        """Run the dashboard application"""
        if not self.storage:
            st.error("Failed to initialize dashboard. Please check the logs.")
            return

        st.title(Config.TITLE)
        st.markdown(Config.DESCRIPTION)

        # Sidebar for experiment selection and controls
        self.setup_sidebar()

        # Main content
        if "selected_experiment" in st.session_state:
            self.display_experiment_analysis()

    def setup_sidebar(self):
        """Setup sidebar controls"""
        st.sidebar.title("Controls")

        # Experiment selection
        experiments = list(self.storage.experiments.keys())
        selected = st.sidebar.selectbox(
            "Select Experiment",
            experiments,
            key="selected_experiment"
        )

        if selected:
            st.session_state.exp = self.storage.experiments[selected]
            st.session_state.df = st.session_state.exp.results_df.copy()

    def display_experiment_analysis(self):
        """Display the main analysis content"""
        df = st.session_state.df

        # Get column types
        config_cols, result_cols = self.processor.get_column_types(df)

        # Calculate statistics
        stats = self.processor.calculate_stats(df)

        # Create tabs for different analyses
        tabs = st.tabs([
            "Overview",
            "Configuration Analysis",
            "Performance Metrics",
            "Clustering",
            "Optimization"
        ])

        # Overview Tab
        with tabs[0]:
            self.display_overview(stats)

        # Configuration Analysis Tab
        with tabs[1]:
            self.display_config_analysis(df, config_cols)

        # Performance Metrics Tab
        with tabs[2]:
            self.display_performance_metrics(df, result_cols)

        # Clustering Tab
        with tabs[3]:
            self.display_clustering_analysis(df, config_cols)

        # Optimization Tab
        with tabs[4]:
            self.display_optimization(df, config_cols, result_cols)

    def display_overview(self, stats: Dict):
        """Display overview statistics and charts"""
        st.header("Overview")

        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Trials", stats["total"])
        col2.metric("Successful Trials", stats["success"])
        col3.metric("Failure Rate", f"{stats['failure_rate']:.1f}%")

        # Display trial outcomes pie chart
        st.plotly_chart(
            self.visualizer.plot_trial_outcomes(stats),
            use_container_width=True
        )

    def display_config_analysis(self, df: pd.DataFrame, config_cols: List[str]):
        """Display configuration analysis"""
        st.header("Configuration Analysis")

        # Select configurations to analyze
        selected_configs = st.multiselect(
            "Select Configuration Parameters",
            config_cols,
            default=config_cols[:2]
        )

        if selected_configs:
            # Display correlation heatmap
            st.plotly_chart(
                self.visualizer.plot_correlation_heatmap(df, selected_configs),
                use_container_width=True
            )

    def display_performance_metrics(self, df: pd.DataFrame, result_cols: List[str]):
        """Display performance metrics analysis"""
        st.header("Performance Metrics")

        # Select metric to analyze
        selected_metric = st.selectbox(
            "Select Metric",
            result_cols
        )

        if selected_metric:
            # Display distribution plot
            st.plotly_chart(
                self.visualizer.plot_metric_distribution(df, selected_metric),
                use_container_width=True
            )

            # Display summary statistics
            st.write("Summary Statistics:")
            st.dataframe(df[selected_metric].describe())

    def display_clustering_analysis(self, df: pd.DataFrame, config_cols: List[str]):
        """Display clustering analysis"""
        st.header("Clustering Analysis")

        # Clustering controls
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        selected_features = st.multiselect(
            "Select Features for Clustering",
            config_cols,
            default=config_cols[:3]
        )

        if selected_features and st.button("Perform Clustering"):
            clustering_results = self.processor.perform_clustering(
                df, selected_features, n_clusters
            )

            # Add cluster labels to dataframe
            df_cluster = df.copy()
            df_cluster['Cluster'] = clustering_results['labels']

            # Display cluster visualization
            if len(selected_features) >= 2:
                fig = px.scatter(
                    df_cluster,
                    x=selected_features[0],
                    y=selected_features[1],
                    color='Cluster',
                    title="Cluster Visualization",
                    template=Config.THEME
                )
                st.plotly_chart(fig, use_container_width=True)

    def display_optimization(self, df: pd.DataFrame, config_cols: List[str], result_cols: List[str]):
        """Display optimization suggestions"""
        st.header("Configuration Optimization")

        # Select target metric
        target_metric = st.selectbox(
            "Select Target Metric for Optimization",
            result_cols
        )

        if target_metric:
            # Find best configuration
            best_idx = df[target_metric].idxmax()
            best_config = df.loc[best_idx, config_cols]

            st.write("### Best Performing Configuration")
            st.write(f"Target Metric Value: {df.loc[best_idx, target_metric]:.2f}")
            st.dataframe(pd.DataFrame([best_config]))

def main():
    """Main entry point"""
    try:
        dashboard = Dashboard()
        dashboard.run()
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import streamlit as st
# import plotly
# import plotly.express as px
# import plotly.graph_objs as go
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from mlos_bench.storage import from_config

# # --------------------------------------------------------------------------------
# # Streamlit Configuration
# # --------------------------------------------------------------------------------
# st.set_page_config(
#     page_title="Azure MySQL Config Analyzer", layout="wide", initial_sidebar_state="expanded"
# )

# # --------------------------------------------------------------------------------
# # Data Loading and Caching
# # --------------------------------------------------------------------------------
# @st.cache_resource
# def load_storage():
#     """
#     Load the MLOS storage configuration for the experiments.
#     This function is cached to prevent reloading on every interaction.
#     """
#     return from_config(config="storage/sqlite.jsonc")

# storage = load_storage()

# # --------------------------------------------------------------------------------
# # Sidebar - Experiment Selection and Filtering
# # --------------------------------------------------------------------------------
# st.sidebar.title("Azure MySQL Config Analyzer")

# # Experiment Selection
# experiment_id = st.sidebar.selectbox(
#     "Select Experiment",
#     options=list(storage.experiments.keys()),
#     help="Choose the experiment you want to analyze.",
# )

# # Load the selected experiment
# exp = storage.experiments[experiment_id]
# df = exp.results_df.copy()

# # Extract configuration and result columns
# config_columns = [col for col in df.columns if col.startswith("config.")]
# result_columns = [col for col in df.columns if col.startswith("result.")]
# metrics = result_columns

# # --------------------------------------------------------------------------------
# # Main Title and Description
# # --------------------------------------------------------------------------------
# st.title(f"Azure MySQL Experiment: {experiment_id}")
# st.write(f"**Description**: {exp.description}")

# # --------------------------------------------------------------------------------
# # Tabs Creation
# # --------------------------------------------------------------------------------
# tabs = st.tabs(
#     [
#         "Dashboard",
#         "Data Overview",
#         "Configurations Analysis",
#         "Failure Analysis",
#         "Correlation Heatmap",
#         "Parallel Coordinates",
#         "Config Params Scatter",
#         "Top & Bottom Configs",
#         "Optimization Suggestions",
#         "Clustering",
#         "Advanced Statistics",
#         "Anomaly Detection",
#         "Save Analysis",
#     ]
# )

# # --------------------------------------------------------------------------------
# # TAB 1: Dashboard
# # --------------------------------------------------------------------------------
# with tabs[0]:
#     st.header("Dashboard")
#     st.write("### Key Metrics Overview")

#     # Calculate key metrics
#     total_trials = len(df)
#     success_trials = df["status"].value_counts().get("SUCCESS", 0)
#     failure_trials = df["status"].value_counts().get("FAILED", 0)
#     success_rate = (success_trials / total_trials) * 100 if total_trials > 0 else 0
#     failure_rate = (failure_trials / total_trials) * 100 if total_trials > 0 else 0

#     # Display key metrics
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Total Trials", total_trials)
#     col2.metric("Successful Trials", success_trials)
#     col3.metric("Failure Rate (%)", f"{failure_rate:.2f}")

#     # Visualization: Success vs Failure
#     fig = px.pie(
#         names=["Success", "Failure"],
#         values=[success_trials, failure_trials],
#         title="Trial Outcomes",
#         color=["Success", "Failure"],
#         color_discrete_map={"Success": "green", "Failure": "red"},
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     # Visualization: Top 5 Metrics
#     st.write("### Top 5 Metrics")
#     top_metrics = df[result_columns].mean().sort_values(ascending=False).head(5)
#     fig_metrics = px.bar(
#         top_metrics,
#         x=top_metrics.index.str.replace("result.", "").str.replace("_", " ").str.title(),
#         y=top_metrics.values,
#         labels={"x": "Metric", "y": "Average Value"},
#         title="Top 5 Average Metrics",
#         color=top_metrics.values,
#         color_continuous_scale="Blues",
#     )
#     st.plotly_chart(fig_metrics, use_container_width=True)

# # --------------------------------------------------------------------------------
# # TAB 2: Data Overview
# # --------------------------------------------------------------------------------
# with tabs[1]:
#     st.header("Data Overview")
#     st.write("Explore experiment data and key statistics.")

#     # Data Filtering
#     with st.expander("Filter Data"):
#         st.subheader("Apply Filters")
#         trial_id_filter = st.text_input(
#             "Filter by Trial ID (comma-separated)", help="Enter trial IDs separated by commas."
#         )
#         status_filter = st.multiselect(
#             "Filter by Status",
#             options=df["status"].unique(),
#             default=df["status"].unique(),
#             help="Select one or more statuses to filter the trials.",
#         )
#         config_filter = st.multiselect(
#             "Filter by Configuration ID",
#             options=df["tunable_config_id"].unique(),
#             default=df["tunable_config_id"].unique(),
#             help="Select one or more configuration IDs to filter the trials.",
#         )

#         if st.button("Apply Filters"):
#             filtered_df = df.copy()
#             if trial_id_filter:
#                 try:
#                     trial_ids = [
#                         int(tid.strip())
#                         for tid in trial_id_filter.split(",")
#                         if tid.strip().isdigit()
#                     ]
#                     filtered_df = filtered_df[filtered_df["trial_id"].isin(trial_ids)]
#                 except ValueError:
#                     st.error("Please enter valid trial IDs separated by commas.")
#             if status_filter:
#                 filtered_df = filtered_df[filtered_df["status"].isin(status_filter)]
#             if config_filter:
#                 filtered_df = filtered_df[filtered_df["tunable_config_id"].isin(config_filter)]
#             st.session_state.filtered_df = filtered_df
#             st.success("Filters applied successfully!")

#     # Display filtered data or original data
#     if "filtered_df" in st.session_state:
#         display_df = st.session_state.filtered_df
#     else:
#         display_df = df

#     if st.checkbox("Show Data Table"):
#         st.dataframe(display_df)
#         st.write("### Descriptive Statistics:")
#         st.write(display_df.describe())

# # --------------------------------------------------------------------------------
# # TAB 3: Configurations Analysis
# # --------------------------------------------------------------------------------
# with tabs[2]:
#     st.header("Configurations Analysis")
#     st.write("Visualize performance metrics across different configurations.")

#     config_id = st.selectbox(
#         "Select Configuration ID",
#         options=df["tunable_config_id"].unique(),
#         help="Choose a configuration to analyze its performance over trials.",
#     )
#     metric = st.selectbox(
#         "Select Metric", options=metrics, help="Choose a performance metric to visualize."
#     )

#     config_data = df[df["tunable_config_id"] == config_id]
#     fig = px.line(
#         config_data,
#         x="trial_id",
#         y=metric,
#         title=f"{metric.replace('result.', '').replace('_', ' ').title()} over Trials for Configuration {config_id}",
#         markers=True,
#         labels={
#             "trial_id": "Trial ID",
#             metric: metric.replace("result.", "").replace("_", " ").title(),
#         },
#         template="plotly_white",
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     # Additional Insights: Moving Average
#     window_size = st.slider(
#         "Select Moving Average Window Size",
#         1,
#         10,
#         3,
#         help="Smooth the metric by applying a moving average.",
#     )
#     config_data[f"{metric}_MA"] = config_data[metric].rolling(window=window_size).mean()
#     fig_ma = px.line(
#         config_data,
#         x="trial_id",
#         y=f"{metric}_MA",
#         title=f"{metric.replace('result.', '').replace('_', ' ').title()} - Moving Average (Window Size={window_size})",
#         markers=True,
#         labels={
#             "trial_id": "Trial ID",
#             f"{metric}_MA": f"{metric.replace('result.', '').replace('_', ' ').title()} (MA)",
#         },
#         template="plotly_white",
#     )
#     st.plotly_chart(fig_ma, use_container_width=True)

# # --------------------------------------------------------------------------------
# # TAB 4: Failure Analysis
# # --------------------------------------------------------------------------------
# with tabs[3]:
#     st.header("Failure Analysis")
#     st.write("Analyze failure rates and trends across trials.")

#     if "status" in df.columns:
#         # Failure Rate Distribution
#         st.subheader("Failure Rate Distribution")
#         failure_counts = df["status"].value_counts()
#         fig_pie = px.pie(
#             values=failure_counts.values,
#             names=failure_counts.index,
#             title="Failure Rate Distribution",
#             color=failure_counts.index,
#             color_discrete_map={"FAILED": "red", "SUCCESS": "green"},
#         )
#         st.plotly_chart(fig_pie, use_container_width=True)

#         # Failure Rate Trend Over Trials
#         st.subheader("Failure Rate Trend Over Trials")
#         failure_rate_trend = (
#             df.groupby("trial_id")["status"]
#             .apply(lambda x: (x == "FAILED").mean() * 100)
#             .reset_index()
#         )
#         failure_rate_trend.columns = ["Trial ID", "Failure Rate (%)"]
#         fig_line = px.line(
#             failure_rate_trend,
#             x="Trial ID",
#             y="Failure Rate (%)",
#             title="Failure Rate Trend Over Trials",
#             markers=True,
#             labels={"Trial ID": "Trial ID", "Failure Rate (%)": "Failure Rate (%)"},
#             template="plotly_white",
#         )
#         st.plotly_chart(fig_line, use_container_width=True)
#     else:
#         st.info("No 'status' column found in the dataset.")

# # --------------------------------------------------------------------------------
# # TAB 5: Correlation Heatmap
# # --------------------------------------------------------------------------------
# with tabs[4]:
#     st.header("Correlation Heatmap")
#     st.write("Visualize correlations between selected configuration and result metrics.")

#     selected_columns = st.multiselect(
#         "Select Columns for Heatmap",
#         options=config_columns + result_columns,
#         default=config_columns[:2] + result_columns[:2],
#         help="Choose multiple columns to analyze their correlation.",
#     )

#     if st.button("Generate Heatmap"):
#         if selected_columns:
#             corr_matrix = df[selected_columns].corr()
#             fig = px.imshow(
#                 corr_matrix,
#                 text_auto=True,
#                 color_continuous_scale="Viridis",
#                 title="Correlation Heatmap",
#                 labels={"color": "Correlation Coefficient"},
#             )
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.warning("Please select at least one column to generate the heatmap.")
#     else:
#         st.info("Select columns and click 'Generate Heatmap' to visualize correlations.")

# # --------------------------------------------------------------------------------
# # TAB 6: Parallel Coordinates
# # --------------------------------------------------------------------------------
# with tabs[5]:
#     st.header("Parallel Coordinates Plot")
#     st.write(
#         "Explore multi-dimensional relationships between configuration parameters and metrics."
#     )

#     parallel_columns = st.multiselect(
#         "Select Columns for Parallel Plot",
#         options=config_columns + result_columns,
#         default=config_columns[:3] + result_columns[:2],
#         help="Choose multiple columns to include in the parallel coordinates plot.",
#     )

#     if parallel_columns:
#         color_metric = st.selectbox(
#             "Select Metric for Coloring",
#             options=result_columns,
#             help="Choose a result metric to color-code the parallel coordinates.",
#         )
#         fig = px.parallel_coordinates(
#             df,
#             dimensions=parallel_columns,
#             color=color_metric,
#             color_continuous_scale=px.colors.diverging.Tealrose,
#             title="Parallel Coordinates Plot",
#             labels={
#                 col: col.replace("config.", "").replace("_", " ").title()
#                 for col in parallel_columns
#             },
#             template="plotly_white",
#         )
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info("Select columns to generate the parallel coordinates plot.")

# # --------------------------------------------------------------------------------
# # TAB 7: Config Params Scatter
# # --------------------------------------------------------------------------------
# with tabs[6]:
#     st.header("Configuration Parameters Scatter Plot")
#     st.write(
#         "Analyze the relationship between multiple configuration parameters and a selected result metric."
#     )

#     if not config_columns:
#         st.warning("No configuration parameters available in the dataset.")
#     elif not metrics:
#         st.warning("No result metrics available in the dataset.")
#     else:
#         # Select multiple configuration parameters
#         selected_config_params = st.multiselect(
#             "Select Configuration Parameters",
#             options=config_columns,
#             default=config_columns[:2],
#             help="Choose one or more configuration parameters to analyze.",
#         )

#         # Select one result metric
#         selected_result_metric = st.selectbox(
#             "Select Result Metric",
#             options=metrics,
#             help="Choose a result metric to analyze against the selected configuration parameters.",
#         )

#         if selected_config_params:
#             # Determine layout based on number of selected parameters
#             plots_per_row = 2
#             num_plots = len(selected_config_params)
#             num_rows = (num_plots + plots_per_row - 1) // plots_per_row

#             for row in range(num_rows):
#                 cols = st.columns(plots_per_row)
#                 for i in range(plots_per_row):
#                     plot_index = row * plots_per_row + i
#                     if plot_index < num_plots:
#                         config_param = selected_config_params[plot_index]
#                         with cols[i]:
#                             fig = px.scatter(
#                                 df,
#                                 x=config_param,
#                                 y=selected_result_metric,
#                                 color="tunable_config_id",
#                                 title=f"{config_param.replace('config.', '').replace('_', ' ').title()} vs {selected_result_metric.replace('result.', '').replace('_', ' ').title()}",
#                                 labels={
#                                     config_param: config_param.replace("config.", "")
#                                     .replace("_", " ")
#                                     .title(),
#                                     selected_result_metric: selected_result_metric.replace(
#                                         "result.", ""
#                                     )
#                                     .replace("_", " ")
#                                     .title(),
#                                 },
#                                 hover_data=["trial_id", "tunable_config_id"],
#                                 trendline="ols",
#                                 template="plotly_white",
#                             )

#                             st.plotly_chart(fig, use_container_width=True)

#                             # Calculate and display the correlation coefficient
#                             corr_coeff = (
#                                 df[[config_param, selected_result_metric]].corr().iloc[0, 1]
#                             )
#                             st.markdown(f"**Correlation Coefficient:** {corr_coeff:.2f}")
#         else:
#             st.info(
#                 "Please select at least one configuration parameter to generate scatter plots."
#             )

# # --------------------------------------------------------------------------------
# # TAB 8: Top & Bottom Configurations
# # --------------------------------------------------------------------------------
# with tabs[7]:
#     st.header("Top and Bottom Configurations")
#     st.write(
#         "Identify configurations with the best and worst performance based on selected metrics."
#     )

#     n_configs = st.slider(
#         "Number of Configurations to Display",
#         min_value=1,
#         max_value=10,
#         value=5,
#         help="Select how many top and bottom configurations to display.",
#     )

#     # Select metric for ranking
#     tb_metric = st.selectbox(
#         "Select Metric for Ranking",
#         options=metrics,
#         index=0,
#         key="tb_metric",
#         help="Choose a metric to rank configurations.",
#     )
#     optimization_method = st.radio(
#         "Select Optimization Method",
#         ["Maximize", "Minimize"],
#         index=0,
#         key="tb_opt_method",
#         help="Choose whether to find configurations that maximize or minimize the selected metric.",
#     )

#     if not df.empty:
#         if optimization_method == "Maximize":
#             top_configs = df.nlargest(n_configs, tb_metric)
#             bottom_configs = df.nsmallest(n_configs, tb_metric)
#         else:
#             top_configs = df.nsmallest(n_configs, tb_metric)
#             bottom_configs = df.nlargest(n_configs, tb_metric)

#         st.subheader("Top Configurations")
#         st.dataframe(top_configs)

#         st.subheader("Bottom Configurations")
#         st.dataframe(bottom_configs)
#     else:
#         st.warning("No data available to identify top/bottom configurations.")

# # --------------------------------------------------------------------------------
# # TAB 9: Optimization Suggestions
# # --------------------------------------------------------------------------------
# with tabs[8]:
#     st.header("Optimization Suggestions")
#     st.write("Discover optimal configurations based on selected performance metrics.")

#     target_metric = st.selectbox(
#         "Select Metric for Optimization",
#         options=metrics,
#         index=0,
#         key="opt_target_metric",
#         help="Choose a performance metric to optimize.",
#     )
#     optimization_method = st.radio(
#         "Select Optimization Method",
#         ["Maximize", "Minimize"],
#         index=0,
#         key="opt_method_choice",
#         help="Choose whether to maximize or minimize the selected metric.",
#     )

#     if not df.empty:
#         if optimization_method == "Maximize":
#             optimal_config = df.loc[df[target_metric].idxmax()]
#         else:
#             optimal_config = df.loc[df[target_metric].idxmin()]

#         st.write(
#             f"**Optimal Configuration ({optimization_method} {target_metric.replace('result.', '').replace('_', ' ').title()}):**"
#         )
#         st.json(optimal_config[config_columns].to_dict())
#     else:
#         st.warning("No data available for optimization.")

# # --------------------------------------------------------------------------------
# # TAB 10: Clustering
# # --------------------------------------------------------------------------------
# with tabs[9]:
#     st.header("Clustering Analysis")
#     st.write("Group similar configurations to identify patterns and clusters.")

#     cluster_columns = st.multiselect(
#         "Select Columns for Clustering",
#         options=config_columns + result_columns,
#         default=config_columns[:3],
#         help="Choose multiple columns to perform clustering.",
#     )
#     num_clusters = st.slider(
#         "Number of Clusters",
#         min_value=2,
#         max_value=10,
#         value=3,
#         help="Define the number of clusters for K-Means.",
#     )

#     if len(cluster_columns) >= 2:
#         if st.button("Generate Clustering"):
#             clustering_data = df[cluster_columns].dropna()

#             # Standardize the data
#             scaler = StandardScaler()
#             clustering_data_scaled = scaler.fit_transform(clustering_data)

#             # Perform K-Means clustering
#             kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#             clusters = kmeans.fit_predict(clustering_data_scaled)
#             df["cluster"] = clusters

#             # Optional: Dimensionality Reduction for 3D Plotting
#             if len(cluster_columns) > 3:
#                 pca = PCA(n_components=3)
#                 principal_components = pca.fit_transform(clustering_data_scaled)
#                 df["PC1"] = principal_components[:, 0]
#                 df["PC2"] = principal_components[:, 1]
#                 df["PC3"] = principal_components[:, 2]
#                 fig = px.scatter_3d(
#                     df,
#                     x="PC1",
#                     y="PC2",
#                     z="PC3",
#                     color="cluster",
#                     title="3D Scatter Plot with PCA and Clustering",
#                     labels={
#                         "PC1": "Principal Component 1",
#                         "PC2": "Principal Component 2",
#                         "PC3": "Principal Component 3",
#                     },
#                     template="plotly_white",
#                 )
#             elif len(cluster_columns) == 3:
#                 fig = px.scatter_3d(
#                     df,
#                     x=cluster_columns[0],
#                     y=cluster_columns[1],
#                     z=cluster_columns[2],
#                     color="cluster",
#                     title="3D Scatter Plot with Clustering",
#                     labels={
#                         cluster_columns[0]: cluster_columns[0]
#                         .replace("config.", "")
#                         .replace("_", " ")
#                         .title(),
#                         cluster_columns[1]: cluster_columns[1]
#                         .replace("config.", "")
#                         .replace("_", " ")
#                         .title(),
#                         cluster_columns[2]: cluster_columns[2]
#                         .replace("config.", "")
#                         .replace("_", " ")
#                         .title(),
#                     },
#                     template="plotly_white",
#                 )
#             else:
#                 fig = px.scatter(
#                     df,
#                     x=cluster_columns[0],
#                     y=cluster_columns[1],
#                     color="cluster",
#                     title="2D Scatter Plot with Clustering",
#                     labels={
#                         cluster_columns[0]: cluster_columns[0]
#                         .replace("config.", "")
#                         .replace("_", " ")
#                         .title(),
#                         cluster_columns[1]: cluster_columns[1]
#                         .replace("config.", "")
#                         .replace("_", " ")
#                         .title(),
#                     },
#                     template="plotly_white",
#                 )

#             st.plotly_chart(fig, use_container_width=True)

#             # Cluster Centroids
#             centroids = kmeans.cluster_centers_
#             centroids_df = pd.DataFrame(centroids, columns=cluster_columns)
#             st.subheader("Cluster Centroids")
#             st.write(centroids_df)
#     else:
#         st.warning("Please select at least two columns for clustering.")

# # --------------------------------------------------------------------------------
# # TAB 10: Advanced Statistics
# # --------------------------------------------------------------------------------
# with tabs[10]:
#     st.header("Advanced Statistics")
#     st.write("Perform advanced statistical analyses on the experiment data.")

#     # Select Metric for Statistical Analysis
#     selected_metric = st.selectbox(
#         "Select Metric for Statistical Analysis",
#         options=metrics,
#         help="Choose a result metric to perform statistical tests.",
#     )

#     # Debugging: Display selected_metric and its type
#     st.write(f"**Selected Metric:** {selected_metric}")
#     st.write(f"**Selected Metric Type:** {df[selected_metric].dtype}")

#     # Check if the selected metric is numeric
#     if pd.api.types.is_numeric_dtype(df[selected_metric]):
#         st.subheader(
#             f"Statistical Summary for {selected_metric.replace('result.', '').replace('_', ' ').title()}"
#         )
#         st.write(df[selected_metric].describe())

#         # Define the template
#         template_value = "plotly_white"
#         st.write(f"**Template Type:** {type(template_value)}, **Value:** {template_value}")

#         # Histogram with KDE
#         try:
#             fig_hist = px.histogram(
#                 df,
#                 x=selected_metric,
#                 nbins=30,
#                 title=f"Distribution of {selected_metric.replace('result.', '').replace('_', ' ').title()}",
#                 marginal="kde",
#                 labels={
#                     selected_metric: selected_metric.replace("result.", "")
#                     .replace("_", " ")
#                     .title()
#                 },
#                 template=template_value,  # Ensure this is a string
#             )
#             st.plotly_chart(fig_hist, use_container_width=True)
#         except Exception as e:
#             st.error(f"An error occurred while generating the histogram: {e}")

#         # Box Plot
#         st.subheader(
#             f"Box Plot for {selected_metric.replace('result.', '').replace('_', ' ').title()}"
#         )
#         try:
#             fig_box = px.box(
#                 df,
#                 y=selected_metric,
#                 points="all",
#                 title=f"Box Plot of {selected_metric.replace('result.', '').replace('_', ' ').title()}",
#                 labels={
#                     selected_metric: selected_metric.replace("result.", "")
#                     .replace("_", " ")
#                     .title()
#                 },
#                 template=template_value,  # Ensure this is a string
#             )
#             st.plotly_chart(fig_box, use_container_width=True)
#         except Exception as e:
#             st.error(f"An error occurred while generating the box plot: {e}")

#         # Violin Plot
#         st.subheader(
#             f"Violin Plot for {selected_metric.replace('result.', '').replace('_', ' ').title()}"
#         )
#         try:
#             fig_violin = px.violin(
#                 df,
#                 y=selected_metric,
#                 box=True,
#                 points="all",
#                 title=f"Violin Plot of {selected_metric.replace('result.', '').replace('_', ' ').title()}",
#                 labels={
#                     selected_metric: selected_metric.replace("result.", "")
#                     .replace("_", " ")
#                     .title()
#                 },
#                 template=template_value,  # Ensure this is a string
#             )
#             st.plotly_chart(fig_violin, use_container_width=True)
#         except Exception as e:
#             st.error(f"An error occurred while generating the violin plot: {e}")
#     else:
#         st.warning(
#             f"The selected metric '{selected_metric}' is not numeric and cannot be plotted."
#         )

#     # Display Plotly Version for Debugging
#     st.subheader("Plotly Version")
#     st.write(f"Plotly version: {plotly.__version__}")

#     # Optional: Display the selected template
#     st.subheader("Template Information")
#     st.write(f"Selected Template: {template_value}")


# # --------------------------------------------------------------------------------
# # TAB 12: Anomaly Detection
# # --------------------------------------------------------------------------------
# with tabs[11]:
#     st.header("Anomaly Detection")
#     st.write("Identify anomalous trials based on selected metrics.")

#     anomaly_metric = st.selectbox(
#         "Select Metric for Anomaly Detection",
#         options=metrics,
#         help="Choose a result metric to perform anomaly detection.",
#     )
#     threshold = st.slider(
#         "Set Anomaly Threshold (Standard Deviations)",
#         min_value=1.0,
#         max_value=5.0,
#         value=3.0,
#         step=0.5,
#         help="Define how many standard deviations away from the mean a data point should be to be considered an anomaly.",
#     )

#     mean_val = df[anomaly_metric].mean()
#     std_val = df[anomaly_metric].std()
#     upper_bound = mean_val + threshold * std_val
#     lower_bound = mean_val - threshold * std_val

#     anomalies = df[(df[anomaly_metric] > upper_bound) | (df[anomaly_metric] < lower_bound)]

#     st.subheader(f"Anomalies in {anomaly_metric.replace('result.', '').replace('_', ' ').title()}")
#     if not anomalies.empty:
#         st.write(f"Total Anomalies Detected: {len(anomalies)}")
#         st.dataframe(anomalies)

#         # Visualization: Scatter Plot Highlighting Anomalies
#         fig_anomaly = px.scatter(
#             df,
#             x="trial_id",
#             y=anomaly_metric,
#             color=df.index.isin(anomalies.index),
#             title=f"Anomaly Detection in {anomaly_metric.replace('result.', '').replace('_', ' ').title()}",
#             labels={
#                 "trial_id": "Trial ID",
#                 anomaly_metric: anomaly_metric.replace("result.", "").replace("_", " ").title(),
#             },
#             color_discrete_map={True: "red", False: "blue"},
#             template="plotly_white",
#         )
#         st.plotly_chart(fig_anomaly, use_container_width=True)
#     else:
#         st.success("No anomalies detected based on the current threshold.")

# # --------------------------------------------------------------------------------
# # TAB 13: Save Analysis Report
# # --------------------------------------------------------------------------------
# with tabs[12]:
#     st.header("Save Analysis Report")
#     st.write("Download a comprehensive analysis report of your experiment.")

#     report_options = st.multiselect(
#         "Select Sections to Include in the Report",
#         options=[
#             "Data Overview",
#             "Configurations Analysis",
#             "Failure Analysis",
#             "Correlation Heatmap",
#             "Parallel Coordinates",
#             "Config Params Scatter",
#             "Top & Bottom Configs",
#             "Optimization Suggestions",
#             "Clustering",
#             "Advanced Statistics",
#             "Anomaly Detection",
#         ],
#         default=[
#             "Data Overview",
#             "Configurations Analysis",
#             "Failure Analysis",
#             "Correlation Heatmap",
#             "Top & Bottom Configs",
#             "Optimization Suggestions",
#         ],
#         help="Choose which sections of the analysis you want to include in the report.",
#     )

#     if st.button("Download Report"):
#         # Generate the report based on selected sections
#         report = f"# Azure MySQL Config Analyzer Report\n\n## Experiment: {experiment_id}\n\n**Description:** {exp.description}\n\n"

#         if "Data Overview" in report_options:
#             report += "## Data Overview\n"
#             report += f"### Descriptive Statistics\n{df.describe().to_markdown()}\n\n"

#         if "Configurations Analysis" in report_options:
#             report += "## Configurations Analysis\n"
#             # Example: Include top configuration analysis
#             top_config = df.loc[
#                 df["result.metric"].idxmax()
#             ]  # Replace 'result.metric' with actual metric if needed
#             report += f"### Optimal Configuration\n{top_config[config_columns].to_dict()}\n\n"

#         if "Failure Analysis" in report_options:
#             report += "## Failure Analysis\n"
#             failure_counts = df["status"].value_counts()
#             report += f"### Failure Rate Distribution\n{failure_counts.to_dict()}\n\n"

#         if "Correlation Heatmap" in report_options:
#             report += "## Correlation Heatmap\n"
#             selected_columns = config_columns + result_columns  # Adjust as needed
#             corr_matrix = df[selected_columns].corr()
#             report += f"### Correlation Matrix\n{corr_matrix.to_markdown()}\n\n"

#         if "Parallel Coordinates" in report_options:
#             report += "## Parallel Coordinates\n"
#             # Example placeholder
#             report += "### Parallel Coordinates Plot was generated in the application.\n\n"

#         if "Config Params Scatter" in report_options:
#             report += "## Configuration Parameters Scatter Plot\n"
#             # Example placeholder
#             report += "### Scatter plots were generated in the application.\n\n"

#         if "Top & Bottom Configs" in report_options:
#             report += "## Top & Bottom Configurations\n"
#             n_configs = st.session_state.get("n_configs_display", 5)
#             tb_metric = st.session_state.get("tb_metric", metrics[0])
#             optimization_method = st.session_state.get("tb_opt_method", "Maximize")
#             if optimization_method == "Maximize":
#                 top_configs = df.nlargest(n_configs, tb_metric)
#                 bottom_configs = df.nsmallest(n_configs, tb_metric)
#             else:
#                 top_configs = df.nsmallest(n_configs, tb_metric)
#                 bottom_configs = df.nlargest(n_configs, tb_metric)
#             report += f"### Top {n_configs} Configurations Based on {tb_metric.replace('result.', '').replace('_', ' ').title()}\n{top_configs.to_markdown()}\n\n"
#             report += f"### Bottom {n_configs} Configurations Based on {tb_metric.replace('result.', '').replace('_', ' ').title()}\n{bottom_configs.to_markdown()}\n\n"

#         if "Optimization Suggestions" in report_options:
#             report += "## Optimization Suggestions\n"
#             target_metric = st.session_state.get("opt_target_metric", metrics[0])
#             optimization_method = st.session_state.get("opt_method_choice", "Maximize")
#             if optimization_method == "Maximize":
#                 optimal_config = df.loc[df[target_metric].idxmax()]
#             else:
#                 optimal_config = df.loc[df[target_metric].idxmin()]
#             report += f"### Optimal Configuration ({optimization_method} {target_metric.replace('result.', '').replace('_', ' ').title()}):\n{optimal_config[config_columns].to_dict()}\n\n"

#         if "Clustering" in report_options:
#             report += "## Clustering Analysis\n"
#             # Example placeholder
#             report += "### Clustering results were generated in the application.\n\n"

#         if "Advanced Statistics" in report_options:
#             report += "## Advanced Statistics\n"
#             selected_metric = st.session_state.get("advanced_stat_metric", metrics[0])
#             report += f"### Statistical Summary for {selected_metric.replace('result.', '').replace('_', ' ').title()}\n{df[selected_metric].describe().to_markdown()}\n\n"

#         if "Anomaly Detection" in report_options:
#             report += "## Anomaly Detection\n"
#             anomaly_metric = st.session_state.get("anomaly_metric", metrics[0])
#             threshold = st.session_state.get("anomaly_threshold", 3.0)
#             mean_val = df[anomaly_metric].mean()
#             std_val = df[anomaly_metric].std()
#             upper_bound = mean_val + threshold * std_val
#             lower_bound = mean_val - threshold * std_val
#             anomalies = df[(df[anomaly_metric] > upper_bound) | (df[anomaly_metric] < lower_bound)]
#             report += f"### Anomalies in {anomaly_metric.replace('result.', '').replace('_', ' ').title()} (Threshold: {threshold} Std Dev)\n{anomalies.to_markdown()}\n\n"

#         # Download the report as a text file
#         st.download_button(
#             label="Download Report as Text",
#             data=report,
#             file_name="analysis_report.txt",
#             mime="text/plain",
#         )

#         # Optionally, provide the CSV report
#         st.subheader("Download Descriptive Statistics")
#         if st.button("Download Descriptive Statistics as CSV"):
#             report_csv = df.describe().to_csv()
#             st.download_button(
#                 label="Download CSV Report",
#                 data=report_csv,
#                 file_name="descriptive_statistics.csv",
#                 mime="text/csv",
#             )

#     st.info("Select the sections you want to include in the report and click 'Download Report'.")

# # --------------------------------------------------------------------------------
# # TAB 10: Clustering
# # --------------------------------------------------------------------------------
# with tabs[9]:
#     st.header("Clustering Analysis")
#     st.write("Group similar configurations to identify patterns and clusters.")

#     cluster_columns = st.multiselect(
#         "Select Columns for Clustering",
#         options=config_columns + result_columns,
#         default=config_columns[:3],
#         help="Choose multiple columns to perform clustering.",
#         key="clustering_columns_select",  # Unique key
#     )

#     num_clusters = st.slider(
#         "Number of Clusters",
#         min_value=2,
#         max_value=10,
#         value=3,
#         help="Define the number of clusters for K-Means.",
#         key="num_clusters_slider_clustering",  # Unique key
#     )

#     if len(cluster_columns) >= 2:
#         if st.button("Generate Clustering", key="gen cluster"):
#             clustering_data = df[cluster_columns].dropna()

#             # Standardize the data
#             scaler = StandardScaler()
#             clustering_data_scaled = scaler.fit_transform(clustering_data)

#             # Perform K-Means clustering
#             kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#             clusters = kmeans.fit_predict(clustering_data_scaled)
#             df["cluster"] = clusters

#             # Optional: Dimensionality Reduction for 3D Plotting
#             if len(cluster_columns) > 3:
#                 pca = PCA(n_components=3)
#                 principal_components = pca.fit_transform(clustering_data_scaled)
#                 df["PC1"] = principal_components[:, 0]
#                 df["PC2"] = principal_components[:, 1]
#                 df["PC3"] = principal_components[:, 2]
#                 fig = px.scatter_3d(
#                     df,
#                     x="PC1",
#                     y="PC2",
#                     z="PC3",
#                     color="cluster",
#                     title="3D Scatter Plot with PCA and Clustering",
#                     labels={
#                         "PC1": "Principal Component 1",
#                         "PC2": "Principal Component 2",
#                         "PC3": "Principal Component 3",
#                     },
#                     template="plotly_white",
#                 )
#             elif len(cluster_columns) == 3:
#                 fig = px.scatter_3d(
#                     df,
#                     x=cluster_columns[0],
#                     y=cluster_columns[1],
#                     z=cluster_columns[2],
#                     color="cluster",
#                     title="3D Scatter Plot with Clustering",
#                     labels={
#                         cluster_columns[0]: cluster_columns[0]
#                         .replace("config.", "")
#                         .replace("_", " ")
#                         .title(),
#                         cluster_columns[1]: cluster_columns[1]
#                         .replace("config.", "")
#                         .replace("_", " ")
#                         .title(),
#                         cluster_columns[2]: cluster_columns[2]
#                         .replace("config.", "")
#                         .replace("_", " ")
#                         .title(),
#                     },
#                     template="plotly_white",
#                 )
#             else:
#                 fig = px.scatter(
#                     df,
#                     x=cluster_columns[0],
#                     y=cluster_columns[1],
#                     color="cluster",
#                     title="2D Scatter Plot with Clustering",
#                     labels={
#                         cluster_columns[0]: cluster_columns[0]
#                         .replace("config.", "")
#                         .replace("_", " ")
#                         .title(),
#                         cluster_columns[1]: cluster_columns[1]
#                         .replace("config.", "")
#                         .replace("_", " ")
#                         .title(),
#                     },
#                     template="plotly_white",
#                 )

#             st.plotly_chart(fig, use_container_width=True)

#             # Cluster Centroids
#             centroids = kmeans.cluster_centers_
#             centroids_df = pd.DataFrame(centroids, columns=cluster_columns)
#             st.subheader("Cluster Centroids")
#             st.write(centroids_df)
#     else:
#         st.warning("Please select at least two columns for clustering.")

# # --------------------------------------------------------------------------------
# # TAB 11: Advanced Statistics
# # --------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------
# # TAB 11: Advanced Statistics
# # --------------------------------------------------------------------------------
# with tabs[10]:
#     st.header("Advanced Statistics")
#     st.write("Perform advanced statistical analyses on the experiment data.")

#     selected_metric = st.selectbox(
#         "Select Metric for Statistical Analysis",
#         options=metrics,
#         help="Choose a result metric to perform statistical tests.",
#         key="sel adv",
#     )

#     st.subheader(
#         f"Statistical Summary for {selected_metric.replace('result.', '').replace('_', ' ').title()}"
#     )

#     # Display data type and missing values
#     st.write(f"Data Type: {df[selected_metric].dtype}")
#     st.write(f"Missing Values: {df[selected_metric].isnull().sum()}")

#     # Handle missing values
#     plot_df = df.dropna(subset=[selected_metric])

#     # Check if the selected metric is numeric
#     if pd.api.types.is_numeric_dtype(plot_df[selected_metric]):
#         st.write(plot_df[selected_metric].describe())

#         # Histogram with KDE
#         fig_hist = px.histogram(
#             plot_df,
#             x=selected_metric,
#             nbins=30,
#             title=f"Distribution of {selected_metric.replace('result.', '').replace('_', ' ').title()}",
#             marginal="kde",
#             labels={
#                 selected_metric: selected_metric.replace("result.", "").replace("_", " ").title()
#             },
#             template="plotly_white",
#         )
#         st.plotly_chart(fig_hist, use_container_width=True)

#         # Box Plot
#         st.subheader(
#             f"Box Plot for {selected_metric.replace('result.', '').replace('_', ' ').title()}"
#         )
#         fig_box = px.box(
#             plot_df,
#             y=selected_metric,
#             points="all",
#             title=f"Box Plot of {selected_metric.replace('result.', '').replace('_', ' ').title()}",
#             labels={
#                 selected_metric: selected_metric.replace("result.", "").replace("_", " ").title()
#             },
#             template="plotly_white",
#         )
#         st.plotly_chart(fig_box, use_container_width=True)

#         # Violin Plot
#         st.subheader(
#             f"Violin Plot for {selected_metric.replace('result.', '').replace('_', ' ').title()}"
#         )
#         fig_violin = px.violin(
#             plot_df,
#             y=selected_metric,
#             box=True,
#             points="all",
#             title=f"Violin Plot of {selected_metric.replace('result.', '').replace('_', ' ').title()}",
#             labels={
#                 selected_metric: selected_metric.replace("result.", "").replace("_", " ").title()
#             },
#             template="plotly_white",
#         )
#         st.plotly_chart(fig_violin, use_container_width=True)
#     else:
#         st.error(
#             f"The selected metric '{selected_metric}' is not numeric. Please select a numeric metric for statistical analysis."
#         )


# # --------------------------------------------------------------------------------
# # TAB 12: Anomaly Detection
# # --------------------------------------------------------------------------------
# with tabs[11]:
#     st.header("Anomaly Detection")
#     st.write("Identify anomalous trials based on selected metrics.")

#     anomaly_metric = st.selectbox(
#         "Select Metric for Anomaly Detection",
#         options=metrics,
#         help="Choose a result metric to perform anomaly detection.",
#     )
#     threshold = st.slider(
#         "Set Anomaly Threshold (Standard Deviations)",
#         min_value=1.0,
#         max_value=5.0,
#         value=3.0,
#         step=0.5,
#         help="Define how many standard deviations away from the mean a data point should be to be considered an anomaly.",
#     )

#     mean_val = df[anomaly_metric].mean()
#     std_val = df[anomaly_metric].std()
#     upper_bound = mean_val + threshold * std_val
#     lower_bound = mean_val - threshold * std_val

#     anomalies = df[(df[anomaly_metric] > upper_bound) | (df[anomaly_metric] < lower_bound)]

#     st.subheader(f"Anomalies in {anomaly_metric.replace('result.', '').replace('_', ' ').title()}")
#     if not anomalies.empty:
#         st.write(f"Total Anomalies Detected: {len(anomalies)}")
#         st.dataframe(anomalies)

#         # Visualization: Scatter Plot Highlighting Anomalies
#         fig_anomaly = px.scatter(
#             df,
#             x="trial_id",
#             y=anomaly_metric,
#             color=df.index.isin(anomalies.index),
#             title=f"Anomaly Detection in {anomaly_metric.replace('result.', '').replace('_', ' ').title()}",
#             labels={
#                 "trial_id": "Trial ID",
#                 anomaly_metric: anomaly_metric.replace("result.", "").replace("_", " ").title(),
#             },
#             color_discrete_map={True: "red", False: "blue"},
#             template="plotly_white",
#         )
#         st.plotly_chart(fig_anomaly, use_container_width=True)
#     else:
#         st.success("No anomalies detected based on the current threshold.")

# # --------------------------------------------------------------------------------
# # TAB 13: Save Analysis Report
# # --------------------------------------------------------------------------------
# with tabs[12]:
#     st.header("Save Analysis Report")
#     st.write("Download a comprehensive analysis report of your experiment.")

#     report_options = st.multiselect(
#         "Select Sections to Include in the Report",
#         options=[
#             "Data Overview",
#             "Configurations Analysis",
#             "Failure Analysis",
#             "Correlation Heatmap",
#             "Parallel Coordinates",
#             "Config Params Scatter",
#             "Top & Bottom Configs",
#             "Optimization Suggestions",
#             "Clustering",
#             "Advanced Statistics",
#             "Anomaly Detection",
#         ],
#         default=[
#             "Data Overview",
#             "Configurations Analysis",
#             "Failure Analysis",
#             "Correlation Heatmap",
#             "Top & Bottom Configs",
#             "Optimization Suggestions",
#         ],
#         help="Choose which sections of the analysis you want to include in the report.",
#     )

#     if st.button("Download Report"):
#         # Generate the report based on selected sections
#         report = f"# Azure MySQL Config Analyzer Report\n\n## Experiment: {experiment_id}\n\n**Description:** {exp.description}\n\n"

#         if "Data Overview" in report_options:
#             report += "## Data Overview\n"
#             report += f"### Descriptive Statistics\n{df.describe().to_markdown()}\n\n"

#         if "Configurations Analysis" in report_options:
#             report += "## Configurations Analysis\n"
#             # Example: Include top configuration analysis
#             if "result.metric" in df.columns:
#                 top_config = df.loc[
#                     df["result.metric"].idxmax()
#                 ]  # Replace 'result.metric' with actual metric
#                 report += f"### Optimal Configuration\n{top_config[config_columns].to_dict()}\n\n"
#             else:
#                 report += (
#                     "### Configurations Analysis details were generated in the application.\n\n"
#                 )

#         if "Failure Analysis" in report_options:
#             report += "## Failure Analysis\n"
#             failure_counts = df["status"].value_counts()
#             report += f"### Failure Rate Distribution\n{failure_counts.to_dict()}\n\n"

#         if "Correlation Heatmap" in report_options:
#             report += "## Correlation Heatmap\n"
#             selected_columns = config_columns + result_columns  # Adjust as needed
#             corr_matrix = df[selected_columns].corr()
#             report += f"### Correlation Matrix\n{corr_matrix.to_markdown()}\n\n"

#         if "Parallel Coordinates" in report_options:
#             report += "## Parallel Coordinates\n"
#             report += "### Parallel Coordinates Plot was generated in the application.\n\n"

#         if "Config Params Scatter" in report_options:
#             report += "## Configuration Parameters Scatter Plot\n"
#             report += "### Scatter plots were generated in the application.\n\n"

#         if "Top & Bottom Configs" in report_options:
#             report += "## Top & Bottom Configurations\n"
#             n_configs = st.session_state.get("n_configs_display", 5)
#             tb_metric = st.session_state.get("tb_metric", metrics[0])
#             optimization_method = st.session_state.get("tb_opt_method", "Maximize")
#             if optimization_method == "Maximize":
#                 top_configs = df.nlargest(n_configs, tb_metric)
#                 bottom_configs = df.nsmallest(n_configs, tb_metric)
#             else:
#                 top_configs = df.nsmallest(n_configs, tb_metric)
#                 bottom_configs = df.nlargest(n_configs, tb_metric)
#             report += f"### Top {n_configs} Configurations Based on {tb_metric.replace('result.', '').replace('_', ' ').title()}\n{top_configs.to_markdown()}\n\n"
#             report += f"### Bottom {n_configs} Configurations Based on {tb_metric.replace('result.', '').replace('_', ' ').title()}\n{bottom_configs.to_markdown()}\n\n"

#         if "Optimization Suggestions" in report_options:
#             report += "## Optimization Suggestions\n"
#             target_metric = st.session_state.get("opt_target_metric", metrics[0])
#             optimization_method = st.session_state.get("opt_method_choice", "Maximize")
#             if optimization_method == "Maximize":
#                 optimal_config = df.loc[df[target_metric].idxmax()]
#             else:
#                 optimal_config = df.loc[df[target_metric].idxmin()]
#             report += f"### Optimal Configuration ({optimization_method} {target_metric.replace('result.', '').replace('_', ' ').title()}):\n{optimal_config[config_columns].to_dict()}\n\n"

#         if "Clustering" in report_options:
#             report += "## Clustering Analysis\n"
#             report += "### Clustering results were generated in the application.\n\n"

#         if "Advanced Statistics" in report_options:
#             report += "## Advanced Statistics\n"
#             selected_metric = st.session_state.get("advanced_stat_metric", metrics[0])
#             report += f"### Statistical Summary for {selected_metric.replace('result.', '').replace('_', ' ').title()}\n{df[selected_metric].describe().to_markdown()}\n\n"

#         if "Anomaly Detection" in report_options:
#             report += "## Anomaly Detection\n"
#             anomaly_metric = st.session_state.get("anomaly_metric", metrics[0])
#             threshold = st.session_state.get("anomaly_threshold", 3.0)
#             mean_val = df[anomaly_metric].mean()
#             std_val = df[anomaly_metric].std()
#             upper_bound = mean_val + threshold * std_val
#             lower_bound = mean_val - threshold * std_val
#             anomalies = df[(df[anomaly_metric] > upper_bound) | (df[anomaly_metric] < lower_bound)]
#             report += f"### Anomalies in {anomaly_metric.replace('result.', '').replace('_', ' ').title()} (Threshold: {threshold} Std Dev)\n{anomalies.to_markdown()}\n\n"

#         # Download the report as a text file
#         st.download_button(
#             label="Download Report as Text",
#             data=report,
#             file_name="analysis_report.txt",
#             mime="text/plain",
#         )

#         # Optionally, provide the CSV report
#         st.subheader("Download Descriptive Statistics")
#         if st.button("Download Descriptive Statistics as CSV"):
#             report_csv = df.describe().to_csv()
#             st.download_button(
#                 label="Download CSV Report",
#                 data=report_csv,
#                 file_name="descriptive_statistics.csv",
#                 mime="text/csv",
#             )

#     st.info("Select the sections you want to include in the report and click 'Download Report'.")

# # --------------------------------------------------------------------------------
# # Additional UI/UX Enhancements
# # --------------------------------------------------------------------------------
# st.sidebar.markdown("---")
# st.sidebar.markdown("#### Tips for Better Workflow")
# st.sidebar.markdown(
#     """
# - **Start with the Dashboard** to get an overview of key metrics.
# - **Use Data Overview** to understand and filter your dataset.
# - **Configurations Analysis** helps visualize specific configuration performances.
# - **Failure Analysis** highlights trial outcomes and trends.
# - **Correlation Heatmap** and **Parallel Coordinates** allow in-depth correlation and multi-dimensional analysis.
# - **Config Params Scatter** plots relationships between configuration parameters and metrics.
# - **Top & Bottom Configs** identify the best and worst-performing configurations.
# - **Optimization Suggestions** provide insights into optimal configurations.
# - **Clustering** groups similar configurations for pattern recognition.
# - **Advanced Statistics** offers detailed statistical analyses of your metrics.
# - **Anomaly Detection** helps identify outliers and unusual trial performances.
# - **Save Analysis** lets you download a comprehensive report of your findings.
#     """
# )
