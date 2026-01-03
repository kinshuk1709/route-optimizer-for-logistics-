try:
    from src.route_optimizer import RouteOptimizer
    print("✅ Import OK")
    
    optimizer = RouteOptimizer("D:\\route\\src\\cities_data.csv")
    print("✅ Optimizer OK")
    print("Cities:", optimizer.cities_df["City"].head().tolist())
    
    routes = optimizer.find_top_3_routes("Delhi", [], "Mumbai")
    print("✅ Routes OK:", len(routes))
    for r in routes:
        print(f"  {r.rank}: {r.route} | {r.distancekm}km")
        
except Exception as e:
    print("❌ ERROR:", str(e))
    import traceback
    traceback.print_exc()
