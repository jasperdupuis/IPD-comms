class Benchmark_Score():
    
    def __init__(self):
        self.benchmark_score = 0
        self.player_score_per_game = []
    
    def set_benchmark_score(self,number_of_turns):
        self.benchmark_score =  number_of_turns * 3 # 3 = player score when player cooperates with opponent
        
    def add_player_score(self,score):
        self.player_score_per_game.append(score)
    
    def get_benchmark_percentage(self):
        return (sum(self.player_score_per_game) * 100) / (len(self.player_score_per_game) * self.benchmark_score)
    
    def rest_scores(self):
        self.player_score_per_game = []
        self.benchmark_score = 0