import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

def compute_meteor_score(reference, candidate):
    candidate_tokens = word_tokenize(candidate)
    reference_tokens = [word_tokenize(reference)]
    score = meteor_score(reference_tokens, candidate_tokens)
    return score

if __name__ == "__main__":
    reference = "well i got my first truck when i was three drove a hundred thousand miles on my knees hauled marbles and rocks and thought twice before i hauled a barbie doll bed for the girl next door she tried to pay me with a kiss and i began to understand there s something women like about a pickup man when i turned sixteen i saved a few hundred bucks my first car was a pickup truck i was cruising the town and the first girl i seen was bobbie jo gentry the homecoming queen she flagged me down and climbed up in the cab and said i never knew you were a pickup man you can set my truck on fire and roll it down a hill and i still wouldn t trade it for a coupe de ville i got an eight foot bed that never has to be made you know if it weren t for trucks we wouldn t have tailgates i met all my wives in traffic jams there s just something women like about a pickup man most friday nights i can be found in the bed of my truck on an old chaise lounge backed into my spot at the drive in show you know a cargo light gives off a romantic glow i never have to wait in line at the popcorn stand cause there s something women like about a pickup man you can set my truck on fire and roll it down a hill and i still wouldn t trade it for a coupe de ville i got an eight foot bed that never has to be made you know if it weren t for trucks we wouldn t have tailgates i met all my wives in traffic jams there s just something women like about a pickup man a bucket of rust or a brand new machine once around the block and you ll know what i mean  you can set my truck on fire and roll it down a hill and i still wouldn t trade it for a coupe de ville i got an eight foot bed that never has to be made you know if it weren t for trucks we wouldn t have tailgates i met all my wives in traffic jams there s just something women like about a pickup man yeah there s something women like about a pickup man"
    temps = [i/10 for i in range(0, 11)]
    for temp in temps:
        filename = "temperature/" + str(int(temp*10)) + ".txt"
        with open(filename, 'r') as file:
            candidate = file.read()
            score = compute_meteor_score(reference, candidate)
            print("meteor score for temp ", temp, ":", score)