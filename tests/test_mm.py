from matchmaker import Matchmaker

Mm = Matchmaker(
    score_file="resources/Bach-fugue_bwv_858.mid",
    performance_file="resources/Bach-fugue_bwv_858.mp3",
    method=
)
for current_position in Mm.run():
    print(current_position);
