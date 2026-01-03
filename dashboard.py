import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import requests
import polyline
from src.route_optimizer import RouteOptimizer
from osrm_distance_calculator import calculate_route_distance

# === NEW: for multi-stop, pdf, traffic ===
import numpy as np
import io
from datetime import datetime

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

# ML Model Integration
try:
    import pickle
    route_model = pickle.load(open('route_cost_model.pkl', 'rb'))
    route_scaler = pickle.load(open('feature_scaler.pkl', 'rb'))
    route_features = pickle.load(open('feature_names.pkl', 'rb'))
    ML_MODEL_AVAILABLE = True
    print("✅ ML Model loaded!")
except:
    ML_MODEL_AVAILABLE = False
    print("⚠️ ML Model not found - using rule-based costs")


st.set_page_config(page_title="🚚 FindMyRoute", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.25rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.12);
}
.metric-card h3 { margin: 0; font-size: 1.8rem; }
.metric-card p { margin: 0.4rem 0 0 0; opacity: 0.9; }

.routebtn button {
    width: 100% !important;
    height: 110px !important;
    border-radius: 14px !important;
    border: 2px solid rgba(102,126,234,0.35) !important;
    background: linear-gradient(135deg, #f7f8fb 0%, #eef2ff 100%) !important;
}
.routebtn-selected button {
    border: 2px solid #667eea !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}
.breakrow {
    display: flex;
    justify-content: space-between;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(15,23,42,0.08);
}
.breakrow:last-child { border-bottom: none; font-weight: 800; }
.toll-badge {
    background: linear-gradient(135deg, #ff4757 0%, #ee5a6f 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-weight: bold;
    display: inline-block;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    cities_df = pd.read_csv("cities_data.csv")
    vehicles_df = pd.read_csv("vehicles_data.csv")
    return cities_df, vehicles_df

def get_city_coords(cities_df, city_name: str):
    row = cities_df[cities_df["City"] == city_name]
    if row.empty:
        return None
    return float(row.iloc[0]["Latitude"]), float(row.iloc[0]["Longitude"])

def get_tolls_for_route(from_city, to_city):
    """Get toll plazas based on route corridor"""
    toll_database = {
        ('Delhi (NCR)', 'Mumbai'): [
            {"name": "Kherki Daula", "lat": 28.4521, "lon": 76.9507, "cost": "₹450", "highway": "NH-48", "km": 42},
            {"name": "Manesar", "lat": 28.3531, "lon": 76.9012, "cost": "₹500", "highway": "NH-48", "km": 50},
            {"name": "Shahjahanpur", "lat": 27.7501, "lon": 75.9501, "cost": "₹350", "highway": "NH-48", "km": 210},
            {"name": "Jaipur", "lat": 26.8983, "lon": 75.7997, "cost": "₹300", "highway": "NH-48", "km": 280},
            {"name": "Kishangarh", "lat": 26.6011, "lon": 74.8532, "cost": "₹250", "highway": "NH-48", "km": 380},
            {"name": "Bharthana", "lat": 22.1008, "lon": 73.4037, "cost": "₹400", "highway": "NH-48", "km": 820},
            {"name": "Shilphata", "lat": 19.0948, "lon": 73.1069, "cost": "₹500", "highway": "NH-48", "km": 1320},
        ],
        ('Delhi (NCR)', 'Bangalore'): [
            {"name": "Ballabgarh", "lat": 28.3629, "lon": 77.3175, "cost": "₹200", "highway": "NH-44", "km": 35},
            {"name": "Agra", "lat": 27.1767, "lon": 78.0081, "cost": "₹250", "highway": "NH-44", "km": 230},
            {"name": "Gwalior", "lat": 26.2183, "lon": 78.1629, "cost": "₹300", "highway": "NH-44", "km": 380},
            {"name": "Bina", "lat": 24.1883, "lon": 78.7668, "cost": "₹200", "highway": "NH-44", "km": 550},
            {"name": "Indore", "lat": 22.7196, "lon": 75.8577, "cost": "₹250", "highway": "NH-44", "km": 750},
            {"name": "Belgaum", "lat": 15.8660, "lon": 75.6237, "cost": "₹300", "highway": "NH-44", "km": 1050},
        ],
        ('Delhi (NCR)', 'Kolkata'): [
            {"name": "Dasna", "lat": 28.6392, "lon": 77.4844, "cost": "₹150", "highway": "NH-6", "km": 45},
            {"name": "Meerut", "lat": 28.9845, "lon": 77.7064, "cost": "₹200", "highway": "NH-6", "km": 65},
            {"name": "Lucknow", "lat": 26.8467, "lon": 80.9462, "cost": "₹300", "highway": "NH-6", "km": 500},
            {"name": "Varanasi", "lat": 25.3200, "lon": 82.9789, "cost": "₹250", "highway": "NH-6", "km": 680},
            {"name": "Patna", "lat": 25.5941, "lon": 85.1376, "cost": "₹200", "highway": "NH-6", "km": 880},
        ],
        ('Mumbai', 'Bangalore'): [
            {"name": "Belgaum", "lat": 15.8660, "lon": 75.6237, "cost": "₹350", "highway": "NH-27", "km": 380},
            {"name": "Hubli", "lat": 15.3647, "lon": 75.1240, "cost": "₹300", "highway": "NH-27", "km": 450},
            {"name": "Chitradurga", "lat": 14.2240, "lon": 75.8826, "cost": "₹250", "highway": "NH-27", "km": 580},
        ],
        ('Delhi (NCR)', 'Ahmedabad'): [
            {"name": "Kherki Daula", "lat": 28.4521, "lon": 76.9507, "cost": "₹200", "highway": "NH-8", "km": 42},
            {"name": "Manesar", "lat": 28.3531, "lon": 76.9012, "cost": "₹250", "highway": "NH-8", "km": 50},
            {"name": "Jaipur", "lat": 26.8983, "lon": 75.7997, "cost": "₹300", "highway": "NH-8", "km": 280},
            {"name": "Ajmer", "lat": 26.4499, "lon": 74.6399, "cost": "₹200", "highway": "NH-8", "km": 380},
        ],
        ('Mumbai', 'Delhi (NCR)'): [
            {"name": "Kherki Daula", "lat": 28.4521, "lon": 76.9507, "cost": "₹450", "highway": "NH-48", "km": 42},
            {"name": "Manesar", "lat": 28.3531, "lon": 76.9012, "cost": "₹500", "highway": "NH-48", "km": 50},
            {"name": "Shahjahanpur", "lat": 27.7501, "lon": 75.9501, "cost": "₹350", "highway": "NH-48", "km": 210},
            {"name": "Jaipur", "lat": 26.8983, "lon": 75.7997, "cost": "₹300", "highway": "NH-48", "km": 280},
            {"name": "Kishangarh", "lat": 26.6011, "lon": 74.8532, "cost": "₹250", "highway": "NH-48", "km": 380},
            {"name": "Bharthana", "lat": 22.1008, "lon": 73.4037, "cost": "₹400", "highway": "NH-48", "km": 820},
            {"name": "Shilphata", "lat": 19.0948, "lon": 73.1069, "cost": "₹500", "highway": "NH-48", "km": 1320},
        ],
        ('Bangalore', 'Delhi (NCR)'): [
            {"name": "Ballabgarh", "lat": 28.3629, "lon": 77.3175, "cost": "₹200", "highway": "NH-44", "km": 35},
            {"name": "Agra", "lat": 27.1767, "lon": 78.0081, "cost": "₹250", "highway": "NH-44", "km": 230},
            {"name": "Gwalior", "lat": 26.2183, "lon": 78.1629, "cost": "₹300", "highway": "NH-44", "km": 380},
            {"name": "Bina", "lat": 24.1883, "lon": 78.7668, "cost": "₹200", "highway": "NH-44", "km": 550},
            {"name": "Indore", "lat": 22.7196, "lon": 75.8577, "cost": "₹250", "highway": "NH-44", "km": 750},
            {"name": "Belgaum", "lat": 15.8660, "lon": 75.6237, "cost": "₹300", "highway": "NH-44", "km": 1050},
        ],
    }

    if (from_city, to_city) in toll_database:
        return toll_database[(from_city, to_city)]
    elif (to_city, from_city) in toll_database:
        return toll_database[(to_city, from_city)]
    else:
        return []

def predict_ml_cost(distance_km, cargo_weight, is_peak=0, is_weekend=0):
    """Predict cost using trained ML model"""
    if not ML_MODEL_AVAILABLE:
        return distance_km * 45 + cargo_weight * 50  # fallback

    features = np.array([[distance_km, cargo_weight, is_peak, is_weekend]])
    features_scaled = route_scaler.transform(features)
    prediction = route_model.predict(features_scaled)[0]
    return prediction

# ===== NEW: Traffic (heuristic) =====
def traffic_multiplier(is_peak: bool, is_weekend: bool) -> float:
    base = 1.0
    if is_peak:
        base += 0.22
    if is_weekend:
        base -= 0.08
    return max(0.85, min(1.35, base))

# ===== NEW: Multi-stop TSP =====
def solve_tsp_order(distance_km_matrix, depot=0):
    n = len(distance_km_matrix)
    manager = pywrapcp.RoutingIndexManager(n, 1, depot)
    routing = pywrapcp.RoutingModel(manager)

    dist_m = (np.array(distance_km_matrix) * 1000).astype(int)

    def dist_cb(from_index, to_index):
        a = manager.IndexToNode(from_index)
        b = manager.IndexToNode(to_index)
        return int(dist_m[a][b])

    transit_cb = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    search = pywrapcp.DefaultRoutingSearchParameters()
    search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search.time_limit.seconds = 3

    solution = routing.SolveWithParameters(search)
    if not solution:
        return None

    index = routing.Start(0)
    order = []
    while not routing.IsEnd(index):
        order.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    order.append(manager.IndexToNode(index))
    return order

@st.cache_data(show_spinner=False)
def compute_distance_matrix_cached(cities_df, city_list):
    n = len(city_list)
    mat = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i][j] = 0.0
            else:
                dist_km, _dur_hr = calculate_route_distance(cities_df, [city_list[i], city_list[j]], use_osrm=True)
                mat[i][j] = float(dist_km)
    return mat

# ===== NEW: PDF Export =====
def build_pdf_bytes(title: str, summary_lines: list[str], table_rows: list[list[str]]):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 0.2*inch))
    for line in summary_lines:
        story.append(Paragraph(line, styles["BodyText"]))
    story.append(Spacer(1, 0.25*inch))

    t = Table(table_rows, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#667eea")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey]),
    ]))
    story.append(t)

    doc.build(story)
    pdf = buf.getvalue()
    buf.close()
    return pdf

def calculate_actual_toll_cost(tolls):
    """Calculate actual toll cost from toll list"""
    total = 0
    for toll in tolls:
        cost_str = toll["cost"].replace("₹", "").strip()
        try:
            total += int(cost_str)
        except:
            pass
    return total

def interpolate_route(lat1, lon1, lat2, lon2, steps=25):
    """Create smooth line between two cities by interpolating points"""
    points = []
    for i in range(steps + 1):
        t = i / steps
        lat = lat1 + (lat2 - lat1) * t
        lon = lon1 + (lon2 - lon1) * t
        points.append((lat, lon))
    return points

def create_proper_route(cities_df, route_cities):
    """Create route with interpolated points between cities (NOT going through sea)"""
    geom = []
    for i in range(len(route_cities) - 1):
        coords_a = get_city_coords(cities_df, route_cities[i])
        coords_b = get_city_coords(cities_df, route_cities[i + 1])
        if coords_a and coords_b:
            segment = interpolate_route(coords_a[0], coords_a[1], coords_b[0], coords_b[1], steps=25)
            if i == 0:
                geom.extend(segment)
            else:
                geom.extend(segment[1:])
    return geom

def osrm_multi_leg(cities_df, route_cities):
    """Build route using OSRM for accurate road distances"""
    all_geom = create_proper_route(cities_df, route_cities)

    total_dist, total_dur = calculate_route_distance(
        cities_df, route_cities, use_osrm=True
    )

    return [], all_geom, total_dist, total_dur, []

def create_highway_table(highways):
    """Create aggregated highway distance table"""
    if not highways:
        return pd.DataFrame()

    highway_dict = {}
    for h in highways:
        hw_name = h["highway"]
        dist = h["distance_km"]
        if hw_name not in highway_dict:
            highway_dict[hw_name] = 0
        highway_dict[hw_name] += dist

    df = pd.DataFrame([
        {"Highway": hw, "Distance (km)": f"{dist:.1f}"}
        for hw, dist in highway_dict.items()
    ])

    return df

cities_df, vehicles_df = load_data()

# ============ SIDEBAR ============
with st.sidebar:
    st.header("🚚 FindMyRoute")

    st.subheader("📍 Trip")
    from_city = st.selectbox("From", cities_df["City"].tolist(), index=0)
    to_city = st.selectbox("To", [c for c in cities_df["City"].tolist() if c != from_city], index=0)

    st.subheader("🚛 Vehicle")
    vehicle_type = st.radio(
        "Select Truck",
        vehicles_df["Vehicle_Type"].tolist(),
        format_func=lambda x: (
            f"{x} • {vehicles_df[vehicles_df.Vehicle_Type==x]['Capacity_Tons'].iloc[0]}T"
            f" • ₹{vehicles_df[vehicles_df.Vehicle_Type==x]['Cost_Per_KM'].iloc[0]:.1f}/km"
        ),
    )

    st.subheader("📦 Cargo")
    cargo_weight = st.slider("Weight (tons)", 1, 25, 10)
    is_fragile = st.checkbox("Fragile (+₹2/km)")
    is_express = st.checkbox("Express (+₹5/km)")

    # ===== NEW: Multi-stop controls =====
    st.markdown("---")
    st.subheader("🧭 Multi-stop (TSP)")
    enable_multistop = st.checkbox("Enable multi-stop planning", value=False)
    extra_stops = []
    if enable_multistop:
        options = [c for c in cities_df["City"].tolist() if c not in [from_city, to_city]]
        extra_stops = st.multiselect("Extra stops (max 5)", options, max_selections=5)

    # ===== NEW: Traffic mode (heuristic) =====
    st.subheader("🚦 Traffic")
    traffic_mode = st.selectbox("Traffic mode", ["Heuristic (free)"], index=0)

    st.divider()

    if st.button("🗺️ Find routes", type="primary", use_container_width=True):

        # ===== NEW: Multi-stop route override =====
        if enable_multistop and len(extra_stops) > 0:
            city_list = [from_city] + extra_stops + [to_city]

            with st.spinner("Computing road distance matrix (OSRM)…"):
                dist_mat = compute_distance_matrix_cached(cities_df, city_list)

            with st.spinner("Solving best stop order (TSP)…"):
                order = solve_tsp_order(dist_mat, depot=0)

            if not order:
                st.error("Could not solve multi-stop route. Try fewer stops.")
                st.stop()

            ordered = [city_list[i] for i in order]

            # OR-Tools route ends back at depot; convert to linear path ending at destination
            if ordered[-1] == from_city:
                ordered = ordered[:-1]
            if ordered[-1] != to_city:
                ordered = [from_city] + [c for c in ordered[1:] if c != to_city] + [to_city]

            _steps, _geom, dist_km, dur_hr, _hwy = osrm_multi_leg(cities_df, ordered)

            now = datetime.now()
            is_peak = int((7 <= now.hour <= 10) or (17 <= now.hour <= 20))
            is_weekend = int(now.weekday() >= 5)

            pred_cost = float(predict_ml_cost(dist_km, cargo_weight, is_peak=is_peak, is_weekend=is_weekend))

            class SimpleRoute:
                def __init__(self, route, total_distance, predicted_cost, estimated_time_hours):
                    self.route = route
                    self.total_distance = total_distance
                    self.predicted_cost = predicted_cost
                    self.estimated_time_hours = estimated_time_hours
                    self.rank = 1
                    self.cost_breakdown = {"ML Predicted Cost": predicted_cost}
                    # ✅ SAFE DEFAULTS so dashboard never crashes
                    self.co2_emissions_kg = 0.0
                    self.safety_plan = ["✅ Safety plan not available in multi-stop mode yet."]
                    self.return_loads = []

            results = [SimpleRoute(ordered, dist_km, pred_cost, dur_hr)]

            st.session_state["results"] = results
            st.session_state["vehicle_type"] = vehicle_type
            st.session_state["cargo_weight"] = cargo_weight
            st.session_state["is_fragile"] = is_fragile
            st.session_state["is_express"] = is_express
            st.session_state["selected_idx"] = 0
            st.session_state["from_city"] = from_city
            st.session_state["to_city"] = to_city
            st.session_state["traffic_mode"] = traffic_mode
            st.rerun()

        # ===== Normal mode (original behavior) =====
        opt = RouteOptimizer("cities_data.csv")
        cap = float(vehicles_df[vehicles_df.Vehicle_Type == vehicle_type]["Capacity_Tons"].iloc[0])
        cpk = float(vehicles_df[vehicles_df.Vehicle_Type == vehicle_type]["Cost_Per_KM"].iloc[0])

        results = opt.find_top_3_routes(
            origin=from_city,
            destination=to_city,
            intermediate_cities=2,
            vehicle_type=vehicle_type,
            vehicle_capacity_tons=cap,
            vehicle_cost_per_km=cpk,
            cargo_weight=cargo_weight,
        )

        st.session_state["results"] = results
        st.session_state["vehicle_type"] = vehicle_type
        st.session_state["cargo_weight"] = cargo_weight
        st.session_state["is_fragile"] = is_fragile
        st.session_state["is_express"] = is_express
        st.session_state["selected_idx"] = 0
        st.session_state["from_city"] = from_city
        st.session_state["to_city"] = to_city
        st.session_state["traffic_mode"] = traffic_mode

        st.rerun()

# ============ MAIN ============
st.title("🚚 Indian Logistics Route Optimizer")
st.caption("Fast, reliable route optimization with clean route visualization")

if "results" not in st.session_state:
    st.info("Select From/To + vehicle in the sidebar, then click **Find routes**.")
    st.stop()

results = st.session_state["results"]
vehicle_type = st.session_state["vehicle_type"]
cargo_weight = st.session_state["cargo_weight"]
is_fragile = st.session_state["is_fragile"]
is_express = st.session_state["is_express"]
from_city = st.session_state.get("from_city", "")
to_city = st.session_state.get("to_city", "")
traffic_mode = st.session_state.get("traffic_mode", "Heuristic (free)")

# ---- Route selector ----
st.subheader("📊 Choose a route")

cols = st.columns(len(results))
for idx, (col, rr) in enumerate(zip(cols, results)):
    surcharge = (rr.total_distance * 2 if is_fragile else 0) + (rr.total_distance * 5 if is_express else 0)
    total_cost = rr.predicted_cost + surcharge

    with col:
        cls = "routebtn-selected" if idx == st.session_state.get("selected_idx", 0) else "routebtn"
        st.markdown(f"<div class='{cls}'>", unsafe_allow_html=True)
        if st.button(
            f"{rr.rank}\n\n📏 {rr.total_distance:.0f} km\n⏱️ {rr.estimated_time_hours:.1f} h\n💰 ₹{total_cost:,.0f}",
            key=f"pick_{idx}",
            use_container_width=True,
        ):
            st.session_state["selected_idx"] = idx
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

sel = results[st.session_state.get("selected_idx", 0)]

# ---- Get route data ----
with st.spinner("📍 Loading map..."):
    steps, geom, osrm_dist_km, osrm_dur_hr, highways = osrm_multi_leg(cities_df, sel.route)

# ---- Tolls ----
tolls = get_tolls_for_route(from_city, to_city)
actual_toll_cost = calculate_actual_toll_cost(tolls)

# ---- Traffic (heuristic) ----
now = datetime.now()
is_peak_now = (7 <= now.hour <= 10) or (17 <= now.hour <= 20)
is_weekend_now = (now.weekday() >= 5)
mult = traffic_multiplier(is_peak=is_peak_now, is_weekend=is_weekend_now)
traffic_eta_hr = osrm_dur_hr * mult
traffic_status = "Heavy" if mult >= 1.2 else ("Moderate" if mult >= 1.05 else "Light")

# ---- Metrics ----
frag = sel.total_distance * 2 if is_fragile else 0
expr = sel.total_distance * 5 if is_express else 0
surcharge = frag + expr
total_cost = sel.predicted_cost + surcharge

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(f"<div class='metric-card'><h3>{sel.total_distance:.0f}</h3><p>km Distance</p></div>", unsafe_allow_html=True)
with m2:
    st.markdown(
        f"<div class='metric-card'><h3>{traffic_eta_hr:.1f}</h3><p>h ETA (Traffic: {traffic_status})</p></div>",
        unsafe_allow_html=True
    )
with m3:
    st.markdown(f"<div class='metric-card'><h3>₹{total_cost:,.0f}</h3><p>Total Cost</p></div>", unsafe_allow_html=True)
with m4:
    # ✅ Works because we ensured defaults in multi-stop mode, and you added it in RouteResult for normal mode
    st.markdown(f"<div class='metric-card'><h3>{getattr(sel, 'co2_emissions_kg', 0.0)}</h3><p>kg CO₂</p></div>", unsafe_allow_html=True)
with m5:
    st.markdown(f"<div class='metric-card'><h3>{len(tolls)}</h3><p>🛑 Toll Plazas</p></div>", unsafe_allow_html=True)

# ---- Route details ----
st.markdown("---")
st.subheader("📍 Route Details")
st.write(f"**Path:** {' → '.join(sel.route)}")
st.write(f"**Vehicle:** {vehicle_type} | **Cargo:** {cargo_weight}T")
st.write(f"**Traffic Mode:** {traffic_mode} | **Multiplier:** x{mult:.2f} | **Status:** {traffic_status}")

# ===== PDF Download =====
st.markdown("---")
st.subheader("📄 Export")
pdf_title = "FindMyRoute - Trip Report"
summary_lines = [
    f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}",
    f"From: {from_city} | To: {to_city}",
    f"Traffic: {traffic_status} (x{mult:.2f})",
]
table_rows = [
    ["Field", "Value"],
    ["Route", " → ".join(sel.route)],
    ["Distance (km)", f"{sel.total_distance:.1f}"],
    ["OSRM Time (hr)", f"{osrm_dur_hr:.1f}"],
    ["Traffic ETA (hr)", f"{traffic_eta_hr:.1f}"],
    ["Predicted Cost (₹)", f"{sel.predicted_cost:,.0f}"],
    ["Total Payable (₹)", f"{total_cost:,.0f}"],
    ["Toll Plazas", str(len(tolls))],
    ["Total Toll Cost (₹)", f"{actual_toll_cost:,.0f}"],
]
pdf_bytes = build_pdf_bytes(pdf_title, summary_lines, table_rows)

st.download_button(
    "⬇️ Download PDF report",
    data=pdf_bytes,
    file_name="route_report.pdf",
    mime="application/pdf",
    use_container_width=True
)

# ---- Cost breakdown ----
st.markdown("---")
st.subheader("💰 Cost Breakdown (Detailed)")

if getattr(sel, "cost_breakdown", None):
    for k, v in sel.cost_breakdown.items():
        st.markdown(f"<div class='breakrow'><span>{k}</span><span>₹{float(v):,.0f}</span></div>", unsafe_allow_html=True)
else:
    st.warning("⚠️ No cost_breakdown in route result")

if is_fragile:
    st.markdown(f"<div class='breakrow'><span>🔨 Fragile surcharge (+₹2/km)</span><span>₹{frag:,.0f}</span></div>", unsafe_allow_html=True)
if is_express:
    st.markdown(f"<div class='breakrow'><span>⚡ Express surcharge (+₹5/km)</span><span>₹{expr:,.0f}</span></div>", unsafe_allow_html=True)

st.markdown(f"<div class='breakrow'><span>💵 Total Payable</span><span>₹{total_cost:,.0f}</span></div>", unsafe_allow_html=True)

# =========================
# ✅ NEW SECTION: SAFETY + BACKHAUL (INSERTED HERE)
# =========================
st.markdown("---")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🛡️ Safety Compliance")
    safety_plan = getattr(sel, "safety_plan", None)
    if safety_plan:
        for s in safety_plan:
            if "REST" in s or "MANDATORY" in s:
                st.warning(s)
            else:
                st.write(s)
    else:
        st.caption("Safety plan not available for this route.")

with col_b:
    st.subheader("💰 Backhaul Opportunity")
    return_loads = getattr(sel, "return_loads", None)
    if return_loads and len(return_loads) > 0:
        load = return_loads[0]
        cargo = load.get("Cargo_Type", "General")
        price = load.get("Price", 0)
        wt = load.get("Weight_Tons", "?")
        load_id = load.get("Load_ID", "-")

        st.info(f"Found Load: {cargo} (₹{price:,})")
        st.caption(f"{to_city} ➝ {from_city} | {wt} tons | Load ID: {load_id}")
    else:
        st.caption("No return load found for this route pair.")

# ============ INTERACTIVE MAP WITH CLEAN ROUTE ============
st.markdown("---")
st.subheader("🗺️ Interactive Map - Route Visualization")

if geom:
    lats = [p[0] for p in geom]
    lons = [p[1] for p in geom]
    center_lat = (max(lats) + min(lats)) / 2
    center_lon = (max(lons) + min(lons)) / 2

    lat_range = max(lats) - min(lats)
    lon_range = max(lons) - min(lons)
    max_range = max(lat_range, lon_range)

    if max_range < 0.1:
        zoom_start = 13
    elif max_range < 0.5:
        zoom_start = 11
    elif max_range < 1:
        zoom_start = 9
    elif max_range < 3:
        zoom_start = 7
    else:
        zoom_start = 5

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True
    )

    folium.PolyLine(
        geom,
        weight=10,
        color="#FF0000",
        opacity=0.95,
        popup="Route Path",
        tooltip="🚚 Main Route"
    ).add_to(m)

    for toll in tolls:
        folium.CircleMarker(
            location=[toll["lat"], toll["lon"]],
            radius=8,
            popup=f"<b>🛑 {toll['name']}</b><br>💰 {toll['cost']}<br>Highway: {toll['highway']}<br>Distance: {toll['km']} km",
            tooltip=f"🛑 {toll['name']} - {toll['cost']}",
            color="#FF0000",
            fill=True,
            fillColor="#FF0000",
            fillOpacity=0.9,
            weight=2
        ).add_to(m)

    for i, city in enumerate(sel.route):
        coords = get_city_coords(cities_df, city)
        if not coords:
            continue

        if i == 0:
            folium.Marker(
                coords,
                popup=f"<b>🚀 START</b><br>{city}",
                tooltip=f"Start: {city}",
                icon=folium.Icon(color="green", icon="play", prefix="fa", icon_color="white")
            ).add_to(m)
        elif i == len(sel.route) - 1:
            folium.Marker(
                coords,
                popup=f"<b>🏁 DESTINATION</b><br>{city}",
                tooltip=f"End: {city}",
                icon=folium.Icon(color="blue", icon="flag", prefix="fa", icon_color="white")
            ).add_to(m)
        else:
            folium.Marker(
                coords,
                popup=f"<b>🔹 STOP</b><br>{city}",
                tooltip=f"Stop: {city}",
                icon=folium.Icon(color="orange", icon="map-pin", prefix="fa", icon_color="white")
            ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, use_container_width=True, height=650)

    st.success("✅ **Route Map** - Clean direct route between cities")
else:
    st.warning("⚠️ Unable to load map data")

# ============ TOLL PLAZAS TABLE ============
st.markdown("---")
st.subheader(f"🛑 Toll Plazas on Route ({len(tolls)} found)")

if tolls:
    toll_df = pd.DataFrame(tolls)
    toll_df_display = toll_df[["name", "highway", "km", "cost"]].copy()
    toll_df_display.columns = ["Toll Plaza", "Highway", "Distance from Start (km)", "Truck Cost"]

    st.dataframe(
        toll_df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Toll Plaza": st.column_config.TextColumn("🛑 Toll Plaza", width="large"),
            "Highway": st.column_config.TextColumn("🛣️ Highway", width="small"),
            "Distance from Start (km)": st.column_config.TextColumn("📏 Distance (km)", width="small"),
            "Truck Cost": st.column_config.TextColumn("💰 Cost", width="small"),
        }
    )

    st.markdown(
        f"<div class='toll-badge'>✅ Total Toll Cost: ₹{actual_toll_cost:,.0f} ({len(tolls)} plazas)</div>",
        unsafe_allow_html=True
    )
else:
    st.warning("⚠️ No toll plazas found for this route")

st.markdown("---")
st.caption("🎯 CartoDB Positron Maps | Direct City Routing | Toll data from NHAI | Production-ready dashboard")
