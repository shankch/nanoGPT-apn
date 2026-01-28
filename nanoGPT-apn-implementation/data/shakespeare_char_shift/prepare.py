"""
Prepare a shifted Shakespeare dataset for continual learning experiments.

This creates Domain B by:
1. Loading the original Shakespeare data and vocabulary (stoi/itos)
2. Appending a synthetic corpus with new character names and patterns
3. Encoding with the SAME vocabulary (stoi/itos) to maintain compatibility

The shift introduces:
- New character names (not in original Shakespeare)
- Different speech patterns and vocabulary emphasis
- Maintains same character set (vocab_size=65)
"""

import os
import pickle
import numpy as np

# First, load the original Shakespeare vocabulary
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
original_meta_path = os.path.join(parent_dir, 'shakespeare_char', 'meta.pkl')

if not os.path.exists(original_meta_path):
    raise FileNotFoundError(
        f"Original meta.pkl not found at {original_meta_path}. "
        "Please run data/shakespeare_char/prepare.py first."
    )

with open(original_meta_path, 'rb') as f:
    original_meta = pickle.load(f)

stoi = original_meta['stoi']
itos = original_meta['itos']
vocab_size = original_meta['vocab_size']

print(f"Loaded vocabulary from original Shakespeare: vocab_size={vocab_size}")
print(f"Characters: {''.join(itos[i] for i in range(vocab_size))}")

def encode(s):
    """Encode string using original vocabulary, skipping unknown chars."""
    return [stoi[c] for c in s if c in stoi]

def decode(l):
    return ''.join([itos[i] for i in l])

# Synthetic corpus with domain shift
# Uses only characters in the original vocab but with different patterns
SYNTHETIC_CORPUS = """
ALDRIC:
Good morrow to thee, friend of ancient days,
The morning star doth rise upon our quest.
What news from distant lands beyond the sea?

BERTRAM:
The kingdoms fall and rise like ocean waves,
And princes scheme while peasants till the soil.
I bring thee tidings dark and full of woe.

ALDRIC:
Speak then, and spare no word of truth or pain,
For I have weathered storms of greater might.
What darkness lurks within thy heavy heart?

BERTRAM:
The northern lords have gathered all their host,
Ten thousand strong they march upon our gates.
Their banners black as night against the dawn.

ALDRIC:
Then we shall meet them with our valiant steel,
And show them that our spirit shall not break.
Call forth the guard and sound the battle horn!

CELESTINE:
Hold fast, my lords, and hear a womans voice,
For war brings naught but widows and their tears.
Perhaps a treaty might preserve our peace?

ALDRIC:
The lady speaks with wisdom in her words,
Yet honor binds us to defend our home.
What say you, Bertram, to this gentle plea?

BERTRAM:
I say we fight, for cowards bend the knee,
And I shall never bow to foreign crowns.
My sword is sharp, my courage shall not fail.

CELESTINE:
Then let it be as fate and gods decree,
But know that I shall pray for your return.
May fortune smile upon your righteous cause.

DAMON:
The ships have sailed from harbors in the east,
Their cargo holds are filled with precious gold.
The merchants speak of wonders yet unseen.

ELARA:
What wonders could surpass our morning sky,
Or match the beauty of a single rose?
I find my treasure in the simple things.

DAMON:
You speak as poets do in gentle rhyme,
But gold can buy us comfort and our peace.
The world runs not on love but on its coin.

ELARA:
Perhaps you speak the truth of worldly ways,
Yet something precious gold can never buy,
The warmth of friendship and a loyal heart.

FELIX:
I interrupt your philosophical debate,
For urgent matters call us to the hall.
The council meets to plan our citys fate.

DAMON:
Then let us go and hear what wisdom says,
For councils oft produce more heat than light.
I hope they do not talk until the dawn.

GABRIELLE:
The gardens bloom with flowers of every hue,
Red roses climb the ancient castle walls.
I tend them daily as my mother did.

HADRIAN:
Your gardens are the envy of the realm,
Each petal placed with care and loving hands.
You have a gift that few can understand.

GABRIELLE:
It is no gift but merely patient work,
The earth rewards those who attend her needs.
Come, let me show you where the lilies grow.

HADRIAN:
I would be honored by such pleasant tour,
For beauty soothes the troubles of the mind.
Lead on, fair lady, to your paradise.

ISADORA:
The winter comes with frost upon the ground,
And bitter winds that howl throughout the night.
We must prepare our stores before the snow.

JASPER:
I have already set the servants task,
The cellars fill with grain and salted meat.
We shall not starve when darkness claims the land.

ISADORA:
You are a prudent man of careful thought,
Such qualities are rare in troubled times.
The household thrives beneath your steady hand.

JASPER:
I merely do what duty does demand,
For many souls depend upon our care.
To fail them would be greater shame than death.

KIERAN:
The scholars gather in the ancient hall,
Their books and scrolls spread wide upon the floor.
They seek the secrets hidden in the past.

LYSANDRA:
What secrets could those dusty pages hold,
That we who live today could not discern?
The present seems enough for mortal minds.

KIERAN:
The ancients knew of things we have forgot,
Of stars that wheel across the endless sky,
Of forces that could shake the very earth.

LYSANDRA:
You speak of magic and of sorcery,
Such things are banned by decree of the crown.
Be careful what you say in public halls.

KIERAN:
I speak of knowledge, not of witchcraft dark,
For truth and magic are not quite the same.
But you are right, I shall be more discreet.

MORGAN:
The feast begins at sunset in the hall,
With music, dancing, and abundant wine.
All nobles of the realm have been invited.

NICHOLAS:
I shall attend in my finest attire,
For such occasions call for proper dress.
Who else shall grace us with their presence there?

MORGAN:
The duke himself shall make an appearance,
Along with lords and ladies of high birth.
It promises to be a grand affair.

NICHOLAS:
Then I must practice my most courtly bow,
And polish my conversational wit.
Such gatherings can make or break a man.

OCTAVIA:
The children play beside the flowing stream,
Their laughter echoes through the summer air.
How innocent they are of worldly cares.

PERCIVAL:
Indeed, youth passes swifter than we know,
And soon enough they too shall bear our burdens.
Let them enjoy these golden days of peace.

OCTAVIA:
You speak with melancholy in your voice,
What troubles weigh upon your noble brow?
Share with me your sorrows and your fears.

PERCIVAL:
I worry for the future of our land,
For shadows gather on the horizon dark.
But let us not disturb the childrens play.

QUINTUS:
The messenger has ridden through the night,
His horse near dead from effort of the journey.
He bears a letter sealed with royal wax.

ROSALIND:
What news could be so urgent and so dire,
That it could not await the morning light?
My heart fills up with worry and with dread.

QUINTUS:
The contents speak of matters of the state,
Which I am not at liberty to share.
The council must convene without delay.

ROSALIND:
Then go, and do what duty does require,
But promise you shall tell me when you can.
I cannot bear this uncertainty alone.

SEBASTIAN:
The tournament begins at break of dawn,
With jousting, archery, and tests of skill.
The finest knights shall compete for glory.

TATIANA:
I shall be there to cheer you from the stands,
With favors made of silk upon my sleeve.
May your lance be true and your shield be strong.

SEBASTIAN:
Your presence gives me courage for the fight,
I shall not fail with you among the crowd.
Watch for my banner flying in the wind.

TATIANA:
I know its colors well, both gold and blue,
And I shall wave whenever you pass by.
Now rest, my love, for morning comes too soon.

ULRIC:
The blacksmith forges weapons for the guard,
The hammer rings from dawn until the dusk.
His arms are thick as branches of an oak.

VIVIENNE:
Such honest labor builds our kingdoms strength,
For without craftsmen we would be undone.
The smith deserves our gratitude and praise.

ULRIC:
I shall convey your words of kind regard,
For he is humble and seeks no reward.
His only joy is metal shaped by fire.

VIVIENNE:
Then bring him to the castle for a feast,
That we might honor him before the court.
Such virtue should not go unrecognized.

WENDELL:
The astronomers have spotted something strange,
A star that moves against the heavens flow.
They say it portends changes yet to come.

XIOMARA:
The sky has always filled men with such wonder,
And fear of what the future might reveal.
I put my faith in action, not in signs.

WENDELL:
A practical philosophy indeed,
Yet even you must wonder at the sight.
Come to the tower and see it for yourself.

XIOMARA:
Perhaps I shall, when duties here are done,
For curiosity is no small thing.
But first I must attend the queens request.

YORICK:
The jesters tale has made the whole court laugh,
His wit as sharp as any warriors blade.
Humor cuts deeper than we often think.

ZELDA:
A fool who makes the powerful laugh aloud,
Has more influence than many a lord.
Never underestimate the jest.

YORICK:
You speak from wisdom born of observation,
For you have watched the court for many years.
What other insights can you share with me?

ZELDA:
Listen more than you speak, and watch with care,
For silence often teaches more than words.
The court is full of masks and hidden blades.

"""

# Repeat and vary the corpus to create more training data
corpus_parts = [SYNTHETIC_CORPUS]

# Add variations with different scene descriptions
SCENE_VARIATIONS = """
ALDRIC:
The sun descends behind the mountain peaks,
And shadows stretch across the valley floor.
Another day has passed in anxious wait.

BERTRAM:
The scouts return with nothing to report,
No sign of movement from the northern host.
Perhaps they hesitate to test our walls.

CELESTINE:
Or perhaps they gather strength for greater strike,
We must not let our guard fall slack from hope.
Vigilance is the price of liberty.

DAMON:
The marketplace is quiet in these times,
Few merchants dare to travel on the roads.
Trade withers like a flower without rain.

ELARA:
The people grow restless and afraid,
They look to us for comfort and for hope.
What words of encouragement can we give?

FELIX:
We tell them truth, that we shall persevere,
That dawn follows even the darkest night.
Our ancestors survived much worse than this.

GABRIELLE:
The healers work from morning until dusk,
For sickness spreads when spirits are so low.
I bring them herbs from my garden beds.

HADRIAN:
Your kindness knows no bounds in troubled times,
The sick and wounded bless your gentle name.
You are an angel walking among us.

ISADORA:
The children ask me when their fathers come,
I have no answer that can ease their pain.
War takes from us more than we ever know.

JASPER:
We must be strong for those who cannot be,
For weakness in the leaders breeds despair.
Put on a brave face for the common folk.

KIERAN:
I have discovered something in the texts,
A passage that may help us in our plight.
The ancients faced a similar dark hour.

LYSANDRA:
What wisdom from the past could aid us now?
Their world was different from the one we know.
Yet speak, for any hope is welcome here.

MORGAN:
The riders come! I see them on the hill!
Their banners show they are our allied lords!
Reinforcements have arrived at last!

NICHOLAS:
Sound the horns and open wide the gates!
Let them enter and refresh their weary bones!
Today our fortune finally has turned!

"""

corpus_parts.append(SCENE_VARIATIONS)

# Combine all parts
full_corpus = "\n".join(corpus_parts)

# Verify all characters are in vocabulary
unknown_chars = set(c for c in full_corpus if c not in stoi)
if unknown_chars:
    print(f"Warning: Unknown characters found: {unknown_chars}")
    print("These will be skipped during encoding.")

# Encode the corpus
encoded_ids = encode(full_corpus)
print(f"\nShifted corpus statistics:")
print(f"  Raw text length: {len(full_corpus):,} characters")
print(f"  Encoded length: {len(encoded_ids):,} tokens")

# Split into train/val (90/10)
n = len(encoded_ids)
train_ids = encoded_ids[:int(n*0.9)]
val_ids = encoded_ids[int(n*0.9):]

print(f"  Train tokens: {len(train_ids):,}")
print(f"  Val tokens: {len(val_ids):,}")

# Save as binary files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

output_dir = os.path.dirname(os.path.abspath(__file__))
train_ids.tofile(os.path.join(output_dir, 'train.bin'))
val_ids.tofile(os.path.join(output_dir, 'val.bin'))

# Save meta with SAME vocabulary as original
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"\nSaved to {output_dir}/")
print("  train.bin")
print("  val.bin") 
print("  meta.pkl (same vocab as shakespeare_char)")
