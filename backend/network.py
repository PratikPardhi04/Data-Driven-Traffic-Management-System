class RoadNetwork:
    def __init__(self):
        self.graph = {
            1: {
                "approaches": {
                    "N": {"downstream": 2},
                    "S": {"downstream": 2},
                    "E": {"downstream": 3},
                    "W": {"downstream": 3}
                }
            },
            2: {},
            3: {}
        }

        self.traffic_state = {}

    def update_traffic(self, intersection_id, approach, density_score):
        if intersection_id not in self.traffic_state:
            self.traffic_state[intersection_id] = {}

        self.traffic_state[intersection_id][approach] = density_score

    def get_pressure(self, intersection_id, approach):
        local_density = self.traffic_state.get(intersection_id, {}).get(approach, 0)

        downstream = self.graph.get(intersection_id, {})\
            .get("approaches", {})\
            .get(approach, {})\
            .get("downstream")

        downstream_density = 0

        if downstream and downstream in self.traffic_state:
            downstream_density = sum(
                self.traffic_state[downstream].values()
            ) / len(self.traffic_state[downstream])

        return local_density - downstream_density
