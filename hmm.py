import numpy as np
from itertools import permutations, chain
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

smoothing_constant = 0.001
in_train_filename = 'twitter_train.txt'

# Create a preprocessing function for Q5(b)
def data_preprocessing(token, tags = None):
    #train_data = data
    preproc_tag = tags
    # Create 2 "columns" -> Token + Preprocessed Tag
    preproc_token_tag = ""

    # Convert all words to lower case
    for word in token:
        if word != "BLANK LINE":
            word.lower()
        else:
            word = "BLANK LINE"

    websites = ["https://", "http://", "www.", ".com"]
    noun_suffix = ("ion", "ment", "ence", "age")
    verb_suffix = ("ing", "ize", "ise", "ed", "ate")
    adj_suffix = ("ful", "ous")
    adv_suffix = ("ly")

    for i in range(len(token)):
        tkn = token[i]
        
        if token[i].startswith("@"):  # "@" Tag
            token[i] = "@@"
            if tags is not None:
                preproc_tag[i] = "@"

        elif token[i].startswith("#"):  # "#" Tag
            token[i] = "##"
            if tags is not None:
                preproc_tag[i] = "#"

        elif any(x in token[i] for x in websites):  # Websites and Emails
            token[i] = "LINK-U"
            if tags is not None:
                preproc_tag[i] = "U"

        elif any(x.isdigit() for x in token[i]):  # Check if it is numeral
            token[i] = "$DIGIT-$"
            if tags is not None:
                preproc_tag[i] = "$"

        elif token[i].endswith(noun_suffix[0]):
            token[i] = "NOUN-SUFFIX"
            if tags is not None:
                preproc_tag[i] = "N"

        elif token[i].endswith(noun_suffix[1]):
            token[i] = "NOUN-SUFFIX"
            if tags is not None:
                preproc_tag[i] = "N"

        elif token[i].endswith(noun_suffix[2]):
            token[i] = "NOUN-SUFFIX"
            if tags is not None:
                preproc_tag[i] = "N"

        elif token[i].endswith(noun_suffix[3]):
            token[i] = "NOUN-SUFFIX"
            if tags is not None:
                preproc_tag[i] = "N"
                
        elif token[i].endswith(verb_suffix[0]):
            token[i] = "VERB-SUFFIX"
            if tags is not None:
                preproc_tag[i] = "V"
        
        elif token[i].endswith(verb_suffix[1]):
            token[i] = "VERB-SUFFIX"
            if tags is not None:
                preproc_tag[i] = "V"

        elif token[i].endswith(verb_suffix[2]):
            token[i] = "VERB-SUFFIX"
            if tags is not None:
                preproc_tag[i] = "V"
        
        elif token[i].endswith(verb_suffix[3]):
            token[i] = "VERB-SUFFIX"
            if tags is not None:
                preproc_tag[i] = "V"

        elif token[i].endswith(verb_suffix[4]):
            token[i] = "VERB-SUFFIX"
            if tags is not None:
                preproc_tag[i] = "V"

        elif token[i].endswith(adj_suffix[0]):
            token[i] = "ADJ-SUFFIX"
            if tags is not None:
                preproc_tag[i] = "A"
        
        elif token[i].endswith(adj_suffix[1]):
            token[i] = "ADJ-SUFFIX"
            if tags is not None:
                preproc_tag[i] = "A"

        elif token[i].endswith(adv_suffix):
            token[i] = "ADV-SUFFIX"
            if tags is not None:
                preproc_tag[i] = "R"

        else:
            token[i] = token[i]
            if tags is not None:
                preproc_tag[i] = preproc_tag[i]

        # Adding to preproc_token_tag
        if token[i] == "BLANK LINE":
            preproc_token_tag += "\n"
        elif tags is None:
            preproc_token_tag += token[i] + '\n'
        else:
            preproc_token_tag += token[i] + '\t' + preproc_tag[i] + '\n'
    return preproc_token_tag

def output_probabilities(in_train_filename, output_file, smoothing, preprocessing=False):
    """
    Estimate output probabilities from train data using maximum likelihood estimation.
    We generate the probabilities through count(y=j->x=w)/count(y=j)
    """
    # Process train data
    with open(in_train_filename) as file:
        pairs = [l.split() for l in file.readlines() if len(l.strip()) != 0]

    if preprocessing == True:
        tokens = []
        tags = []
        for token, tag in pairs:
            tokens.append(token)
            tags.append(tag)
        data = data_preprocessing(tokens, tags)
        pairs = [l.split("\t") for l in data.split("\n")]
        pairs = pairs[:-1]

    token_tag_matrix = {}
    tags_matrix = {}
    tokens = []

    for token, tag in pairs:
        tokens.append(token)
        token_tag = token + '\t' + tag
        if token_tag in token_tag_matrix:
            token_tag_matrix[token_tag] += 1
        else:
            token_tag_matrix[token_tag] = 1

        if tag in tags_matrix:
            tags_matrix[tag] += 1
        else:
            tags_matrix[tag] = 1

    unique_tokens = set(tokens)

    for tag in tags_matrix:
        token_tag_matrix['unseen' + '\t' + tag] = 0

    all_print = ''
    for pair, count in token_tag_matrix.items():
        token = pair.split()[0]
        tag = pair.split()[1]

        probability = (count + smoothing) / (tags_matrix[tag] + smoothing * (len(unique_tokens) + 1))

        all_print += pair + '\t' + str(probability) + "\n"

    print(all_print, file=open(output_file, "w"))

output_probabilities(in_train_filename, "naive_output_probs.txt", smoothing_constant, preprocessing=False)
output_probabilities(in_train_filename, "output_probs.txt",smoothing_constant, preprocessing=False)
output_probabilities(in_train_filename, "output_probs2.txt", 0.001, preprocessing=True)


def mle_output_probs(output_probs_filename, train_filename, tags_filename, smoothing, preprocessing=False):
    """
    Compute transition probabilities from train data using maximum likelihood estimation.
    We generate the probabilities through count(yt-1=i,yt=j)/count(yt-1=i)
    """
    # Process states data
    with open(tags_filename) as f:
        tags = f.readlines()
    tags = [i.strip("\n") for i in tags]

    # Process train data
    with open(train_filename) as f:
        train = f.readlines()
    train = [i.strip("\n").split("\t") for i in train]
    train_words = [w for w, j in list(filter(lambda x: x != [''], train))]

    if preprocessing == True:
        tokens = []
        preproc_tags = []
        for t in train:
            if t == ['']:
                tokens.append("BLANK LINE")
                preproc_tags.append("BLANK LINE")
            else:
                tokens.append(t[0])
                preproc_tags.append(t[1])
        data = data_preprocessing(tokens, preproc_tags)
        data = data[:-1]
        train = [l.split("\t") for l in data.split("\n")]

    # Process transition of states in train
    train_states = ['START']
    for i in range(len(train)):
        if len(train[i]) == 1:
            train_states.append('END')
            if i != len(train) - 1:
                train_states.append('START')
        else:
            train_states.append(train[i][1])

    # P  - permutations of states i -> states j 
    # P1 - permuations between states i
    # P2 - permuatations of start to state i
    # P3 - permuatations of state i to end
    # P4 - permuatations of same state
    P1 = [*permutations(tags, 2)]
    P2 = [*(("START", i) for i in tags)]
    P3 = [*((i, "END") for i in tags)]
    P4 = [*((i, i) for i in tags)]
    P = [*chain(P1, P2, P3, P4)]

    # Counting yt-1=i -> yt=j 
    trans_dict = {}
    for pm in P:
        if pm not in trans_dict:
            trans_dict[pm] = 0

    for i in range(len(train_states)):
        curr_state = train_states[i]
        if curr_state == 'END':
            continue
        next_state = train_states[i + 1]
        trans_dict[(curr_state, next_state)] += 1

    output_list = []
    output_file = open(output_probs_filename, "w")

    # Computing transition probability using MLE
    for trans, count in trans_dict.items():
        curr_state = trans[0]
        numer = count + smoothing
        denom = train_states.count(curr_state) + (smoothing * (len(set(train_words)) + 1))
        trans_prob = numer / denom
        output_list.append(f"{trans[0]}\t{trans[1]}\t{trans_prob}\n")

    # Save output
    output_file.writelines(output_list)

mle_output_probs('trans_probs.txt', 'twitter_train.txt', 'twitter_tags.txt', 0.1, preprocessing=False)
mle_output_probs('trans_probs2.txt', 'twitter_train.txt', 'twitter_tags.txt', 0.001, preprocessing=True)

# Implement the six functions below
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    with open(in_output_probs_filename) as file:
        lines = [l.split() for l in file.readlines() if len(l.strip()) != 0]

    dict = {}
    for token, tag, prob in lines:
        if token not in dict:
            dict[token] = {}
            dict[token][tag] = prob
        else:
            dict[token][tag] = prob

    with open(in_test_filename) as file:
        tokens = [l.strip() for l in file.readlines() if len(l.strip()) != 0]

    all_print = ''

    for token in tokens:
        prob = 0
        chosen_tag = ''

        if token in dict.keys():
            for tag, tag_prob in dict[token].items():
                if float(tag_prob) > prob:
                    prob = float(tag_prob)
                    chosen_tag = tag
            all_print += chosen_tag + "\n"
        else:
            for tag, tag_prob in dict['unseen'].items():
                if float(tag_prob) > prob:
                    prob = float(tag_prob)
                    chosen_tag = tag
            all_print += chosen_tag + "\n"

    print(all_print, file=open(out_prediction_filename, "w"))


def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    tokens_train = {}
    tags_train = {}

    with open(in_train_filename) as file:
        pairs = [l.split() for l in file.readlines() if len(l.strip()) != 0]

    for token, tag in pairs:
        if token in tokens_train:
            tokens_train[token] += 1
        else:
            tokens_train[token] = 1

        if tag in tags_train:
            tags_train[tag] += 1
        else:
            tags_train[tag] = 1

    with open(in_output_probs_filename) as file:
        lines = [l.split() for l in file.readlines() if len(l.strip()) != 0]

    dict = {}
    for token, tag, prob in lines:
        if token not in dict:
            dict[token] = {}
            dict[token][tag] = prob
        else:
            dict[token][tag] = prob

    with open(in_test_filename) as file:
        tokens_test = [l.strip() for l in file.readlines() if len(l.strip()) != 0]

    all_print = ''
    for token in tokens_test:
        prob = 0
        chosen_tag = ''

        if token in dict.keys():
            for tag, prob_x_given_y in dict[token].items():
                prob_y = tags_train[tag] / sum(tags_train.values())
                prob_x = tokens_train[token] / sum(tokens_train.values())
                prob_y_given_x = (float(prob_x_given_y) * prob_y) / prob_x

                if prob_y_given_x > prob:
                    prob = prob_y_given_x
                    chosen_tag = tag
            all_print += chosen_tag + '\n'
        else:
            for tag, prob_x_given_y in dict['unseen'].items():
                prob_y = tags_train[tag] / sum(tags_train.values())
                prob_x = 1  
                prob_y_given_x = (float(prob_x_given_y) * prob_y) / prob_x

                if prob_y_given_x > prob:
                    prob = prob_y_given_x
                    chosen_tag = tag
            all_print += chosen_tag + '\n'
    print(all_print, file=open(out_prediction_filename, "w"))


def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    # Process states data
    with open(in_tags_filename) as f:
        tags = f.readlines()
    tags = [i.strip("\n") for i in tags]

    # Process transition probabilities
    with open(in_trans_probs_filename) as f:
        trans_probs = f.readlines()
    trans_probs = list(filter(lambda x: x != '\n', trans_probs))
    trans_probs = [i.strip("\n").split("\t") for i in trans_probs]
    trans_probs = dict([*map(lambda x: ((x[0], x[1]), float(x[2])), trans_probs)])

    # Process output probabilities
    with open(in_output_probs_filename) as f:
        output_probs = f.readlines()
    output_probs = list(filter(lambda x: x != '\n', output_probs))
    output_probs = [i.strip("\n").split('\t') for i in output_probs]
    train_words = [i[0] for i in output_probs]
    output_probs = dict([*map(lambda x: ((x[0], x[1]), float(x[2])), output_probs)])

    # Process test dataset
    with open(in_test_filename) as f:
        test = f.read()
    test = test.split('\n\n')
    test = list(filter(lambda x: x != '', test))
    test = [i.split("\n") for i in test]

    # Handle output
    output_list = []
    output_file = open(out_predictions_filename, "w")

    # Viterbi algorithm
    # t - tweet, represents a sequence of tokens 
    for t in test:
        # N - number of states
        # n - number of observations
        N = len(tags)
        n = len(t)
        pi = np.zeros((N, n + 1))

        # First loop, START -> state i for first token
        start_token = t[0]
        for i in range(len(tags)):
            if (start_token, tags[i]) not in output_probs:
                prob = trans_probs[("START", tags[i])] * output_probs[('unseen', tags[i])]
            else:
                prob = trans_probs[("START", tags[i])] * output_probs[(start_token, tags[i])]
            pi[i, 0] = prob

        backpointer = np.zeros((N, n - 1))

        # Second loop, for 2nd to nth token  
        for i in range(1, len(t)):

            # state k -> state j for ith token
            curr_token = t[i]
            for j in range(len(tags)):
                temp_product = [trans_probs[(tags[k], tags[j])] * pi[k][i - 1] for k in range(len(tags))]
                if (curr_token, tags[j]) not in output_probs:
                    helper = list(map(lambda x: x * output_probs[('unseen', tags[j])], temp_product))
                else:
                    helper = list(map(lambda x: x * output_probs[(curr_token, tags[j])], temp_product))

                pi[j, i] = max(helper)
                backpointer[j, i - 1] = np.argmax(helper)

        # Third loop, state i -> END for last token
        end_token = t[-1]
        for i in range(len(tags)):
            if (end_token, tags[i]) not in output_probs:
                prob = trans_probs[(tags[i], "END")] * output_probs[('unseen', tags[i])]
            else:
                prob = trans_probs[(tags[i], "END")] * output_probs[(end_token, tags[i])]
            pi[i, -1] = prob

        # Traverse backwards using backpointer
        optimal_tags = np.zeros(n).astype(np.int32)
        optimal_tags[-1] = np.argmax(pi[:, -1])

        for n in range(n - 2, -1, -1):
            optimal_tags[n] = backpointer[optimal_tags[n + 1], n]

        # Handle output
        for i in optimal_tags:
            output_list.append(f"{tags[i]}\n")
        output_list.append("\n")

    # Save output
    output_file.writelines(output_list)

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    # Process states data
    with open(in_tags_filename) as f:
        tags = f.readlines()
    tags = [i.strip("\n") for i in tags]

    # Process transition probabilities
    with open(in_trans_probs_filename) as f:
        trans_probs = f.readlines()
    trans_probs = list(filter(lambda x: x != '\n', trans_probs))
    trans_probs = [i.strip("\n").split("\t") for i in trans_probs]
    trans_probs = dict([*map(lambda x: ((x[0], x[1]), float(x[2])), trans_probs)])

    # Process output probabilities
    with open(in_output_probs_filename) as f:
        output_probs = f.readlines()
    output_probs = list(filter(lambda x: x != '\n', output_probs))
    output_probs = [i.strip("\n").split("\t") for i in output_probs]
    train_words = [i[0] for i in output_probs]
    output_probs = dict([*map(lambda x: ((x[0], x[1]), float(x[2])), output_probs)])
    #print(output_probs[('tears', 'N')])

    # Process test dataset
    with open(in_test_filename) as f:
        test = f.read()
    test = test.split('\n\n')
    test = list(filter(lambda x: x != '', test))
    test = [i.split("\n") for i in test]

    # Data preprocessing
    flatten_test = []
    for sublist in test:
        for i in sublist:
            flatten_test.append(i)
        flatten_test.append("\t")

    data = data_preprocessing(flatten_test, tags = None)
    data = data.split("/n/n")

    new_test = []
    for i in data:
        new_test.append(i.split("\n"))

    new_test = new_test[0]
    new_test = [x.split('//t') for x in '//t'.join(new_test).split('//t\t//t')]
    new_test = new_test[:-1]

    # Handle output
    output_list = []
    output_file = open(out_predictions_filename, "w")

    # Viterbi algo:
    # t - tweet, represents a sequence of tokens
    for t in new_test:
        # N - number of states
        # n - number of observations
        N = len(tags)
        n = len(t)
        pi = np.zeros((N, n + 1))

        # First loop, START -> state i for first token
        start_token = t[0]
        for i in range(len(tags)):
            if (start_token, tags[i]) not in output_probs:
                prob = trans_probs[("START", tags[i])] * output_probs[('unseen', tags[i])]
            else:
                prob = trans_probs[("START", tags[i])] * output_probs[(start_token, tags[i])]
            pi[i, 0] = prob

        backpointer = np.zeros((N, n - 1))

        # Second loop, for 2nd to nth token  
        for i in range(1, len(t)):

            # state k -> state j for ith token
            curr_token = t[i]
            for j in range(len(tags)):
                temp_product = [trans_probs[(tags[k], tags[j])] * pi[k][i - 1] for k in range(len(tags))]
                if (curr_token, tags[j]) not in output_probs:
                    helper = list(map(lambda x: x * output_probs[('unseen', tags[j])], temp_product))
                else:
                    helper = list(map(lambda x: x * output_probs[(curr_token, tags[j])], temp_product))

                pi[j, i] = max(helper)
                backpointer[j, i - 1] = np.argmax(helper)
    
        # Third loop, state i -> END for last token
        end_token = t[-1]
        for i in range(len(tags)):
            if (end_token, tags[i]) not in output_probs:
                prob = trans_probs[(tags[i], "END")] * output_probs[('unseen', tags[i])]
            else:
                prob = trans_probs[(tags[i], "END")] * output_probs[(end_token, tags[i])]
            pi[i, -1] = prob

        # Traverse backwards using backpointer
        optimal_tags = np.zeros(n).astype(np.int32)
        optimal_tags[-1] = np.argmax(pi[:, -1])

        for n in range(n - 2, -1, -1):
            optimal_tags[n] = backpointer[optimal_tags[n + 1], n]

        # Handle output
        for i in optimal_tags:
            output_list.append(f"{tags[i]}\n")
        output_list.append("\n")

    # Save output
    output_file.writelines(output_list)

def forward(start_trans, trans_matrix, output_matrix, states, sentence):
    """
    start_trans: transition probs for start -> state (N)
    trans_matrix: transition probs for prev state -> current state (N * N)
    output_matrix: output probs for state, token (N * n)
    states: states excluding START and END
    sentence: sequence of tokens
    """
    N = len(states)
    n = len(sentence)
    alpha = np.zeros((N, n))
    alpha[:, 0] = start_trans * output_matrix[:, 0]

    for i in range(1, n):
        for j in range(N):
            temp = alpha[:, i-1] * trans_matrix[:, j] * output_matrix[j, i]
            alpha[j, i] = sum(temp)
    
    return alpha

def backward(stop_trans, trans_matrix, output_matrix, states, sentence):
    """
    stop_trans: transition probs for state -> stop
    trans_matrix: transition probs for prev state -> current state (N * N)
    output_matrix: output probs for state, token (N * n)
    states: states excluding START and END
    sentence: sequence of tokens
    """
    N = len(states)
    n = len(sentence)
    beta = np.zeros((N, n))
    beta[:, -1] = stop_trans

    for i in range(n-2, -1, -1):
        for j in range(N):
            temp = trans_matrix[j, :] * output_matrix[:, i+1] * beta[:, i+1]
            beta[j, i] = sum(temp)

    return beta

def get_gamma(alpha, beta, alpha_stop):
    gamma_numerator = alpha * beta
    return gamma_numerator / sum(alpha_stop)

def get_xi(alpha, beta, alpha_stop, output_matrix, trans_matrix, sentence, states):
    """
    alpha: N * N
    beta: N * n
    alpha_stop: N
    output_matrix: output probs for state, token (N * n)
    trans_matrix: transition probs for prev state -> current state (N * N)
    sentence: sequence of tokens
    states: states excluding START and END
    """
    n = len(sentence)
    N = len(states)
    xi = np.zeros((N, N, n-1))

    for n in range(n-1):
        for i in range(N):
            for j in range(N):
                temp_numerator = alpha[i, n] * trans_matrix[i, j] * output_matrix[j, n+1] * beta[j, n+1]
                xi[i,j,n] = temp_numerator / sum(alpha_stop)

    return xi


def update_a(a_numerator,xi):
    # sum xi across time
    a_numerator_temp = np.sum(xi, axis=2)

    # update a_numerator
    a_numerator = [a + b for a, b, in zip(a_numerator, a_numerator_temp)]
    return a_numerator


def update_b(b_numerator,b_denominator,sentence,unique_words,gamma):
    # sum of all gammas where token = v,state = j 
    for i in range(len(sentence)):
        token_index = unique_words.index(sentence[i])
        b_numerator[:, token_index] += gamma[:, i]

    # sum gamma across time
    b_denominator_temp = np.sum(gamma, axis=1)

    # update b_denominator
    b_denominator = [a + b for a, b, in zip(b_denominator, b_denominator_temp)]
    return b_numerator, b_denominator

def update_start_trans(new_start_trans, start_trans, output_matrix, beta, alpha_stop):
    new_start_trans_temp = start_trans * output_matrix[:, 0] * beta[:,0] / sum(alpha_stop)
    new_start_trans = [a + b for a, b, in zip(new_start_trans, new_start_trans_temp)]
    return new_start_trans

def update_stop_trans(new_stop_trans, alpha_stop):
    new_stop_trans_temp = alpha_stop / sum(alpha_stop)
    new_stop_trans = [a + b for a, b, in zip(new_stop_trans, new_stop_trans_temp)]
    return new_stop_trans

def get_log_likelihood(train, start_trans, stop_trans, trans_matrix, output_matrix, states, unique_words):
    log_likelihood = []
    N = len(states)

    for sentence in train:
        n = len(sentence)
        output_matrix_sentence = np.zeros((N, n))

        for i in range(n):
            token_index = unique_words.index(sentence[i])
    
            for j in range(N):
                output_matrix_sentence[j, i] = output_matrix[j, token_index]
        
        alpha = forward(start_trans, trans_matrix, output_matrix_sentence, states, sentence)
        alpha_stop = alpha[:, -1] * stop_trans
        log_likelihood.append(np.log(sum(alpha_stop)))

    return sum(log_likelihood)

def forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh):
    # Process train data
    with open(in_train_filename) as f:
        train = f.read()
    train = train.split('\n\n')
    train = list(filter(lambda x: x != '', train))
    train = [i.split("\n") for i in train]

    # Process unique observations
    with open(in_train_filename) as f:
        obs = f.readlines()
    obs = [i.strip("\n") for i in obs]
    unique_words = list(dict.fromkeys([word for word in obs if word != '']))
    unique_words.append("unseen")

    # Process states data
    with open(in_tag_filename) as f:
        states = f.readlines()
    states = [i.strip("\n") for i in states]

    # Generate transition permutations
    P1 = [*permutations(states, 2)]
    P2 = [*(("START", i) for i in states)]
    P3 = [*((i, "END") for i in states)]
    P4 = [*((i, i) for i in states)]
    P = [*chain(P1, P2, P3, P4)]

    # Generate randomised trans_probs
    trans_dict = {}
    np.random.seed(seed)
    random_p = np.random.random(len(P))
    for i, trans in enumerate(P):
        trans_dict[trans] = random_p[i]

    trans_probs = {}
    for trans, value in trans_dict.items():
        trans_probs[trans] = value / sum([v for k, v in trans_dict.items() if k[0] == trans[0]])

    # Generate randomised output_probs
    output_dict = {}
    output_probs = {}
    np.random.seed(seed)
    for state in states:
        counter = 0
        random_p2 = np.random.random(len(unique_words))
        random_p2 = random_p2 / np.sum(random_p2)
        for word in unique_words:
            key1 = word + "\t" + state
            key2 = (word, state)
            output_dict[key1] = random_p2[counter]
            output_probs[key2] = random_p2[counter]
            counter += 1

    # # Output initialised probs into trans_probs3.txt
    # output_trans = []
    # output_trans_file = open("trans_probs3.txt", "w")
    
    # for k, v in trans_probs.items():
    #     output_trans.append(f"{k[0]}\t{k[1]}\t{v}\n")
    
    # # output initialised probs into output_probs3.txt
    # out_output = open("output_probs3.txt", "w")
    # for item in output_dict.items():
    #     out_output.write(item[0] + "\t" + str(item[1]) + "\n")
    
    # # Save output
    # output_trans_file.writelines(output_trans)

    # Convert trans_probs into array of arrays
    N = len(states)
    start_trans = np.zeros(N)
    stop_trans = np.zeros(N)
    trans_matrix = np.zeros((N, N))
    for k,v in trans_probs.items():
        if k[0] == 'START':
            start_trans[states.index(k[1])] = v
        elif k[1] == 'END':
            stop_trans[states.index(k[0])] = v
        else:
            prev_index = states.index(k[0])
            curr_index = states.index(k[1])
            trans_matrix[prev_index, curr_index] = v

    # Convert output_probs into array of arrays
    l = len(unique_words)
    output_matrix = np.zeros((N, l))
    for item in output_dict.items():
        token = item[0].strip("\t").split()[0]
        tag = item[0].strip("\t").split()[1]
        if token != "unseen":
            token_index = unique_words.index(token)
            tag_index = states.index(tag)
            output_matrix[tag_index, token_index] = item[1]
        else:
            tag_index = states.index(tag)
            output_matrix[tag_index, -1] = item[1]

    log_likelihoods = np.zeros(max_iter+1)
    for iteration in range(1, max_iter+1):
        print("start iteration", iteration)

        # Initialise iteration variables
        new_a_numerator = np.zeros((N, N))
        new_b_numerator = np.zeros((N, l))
        new_b_denominator = np.zeros(N)
        new_start_trans = np.zeros(N)
        new_stop_trans = np.zeros(N)

        for sentence in train:
            n = len(sentence)

            # Generate output_matrix for current sentence only
            output_matrix_sentence = np.zeros((N, n))
            for i in range(n):
                token_index = unique_words.index(sentence[i])
                for j in range(N):
                    output_matrix_sentence[j, i] = output_matrix[j, token_index]

            # Forward probability
            alpha = forward(start_trans, trans_matrix, output_matrix_sentence, states, sentence)
            alpha_stop = alpha[:, -1] * stop_trans

            # Backward probability
            beta = backward(stop_trans, trans_matrix, output_matrix_sentence, states, sentence)

            # E-step
            xi = get_xi(alpha, beta, alpha_stop, output_matrix_sentence, trans_matrix, sentence, states)
            gamma = get_gamma(alpha, beta, alpha_stop)
            
            # Caculate a & b for M-step
            new_a_numerator = update_a(new_a_numerator, xi)
            new_b_numerator, new_b_denominator  = update_b(new_b_numerator, new_b_denominator, sentence, unique_words, gamma)

            # Update new trans_probs and output_probs for next iteration
            new_start_trans = update_start_trans(new_start_trans, start_trans, output_matrix_sentence, beta, alpha_stop)
            new_stop_trans = update_stop_trans(new_stop_trans, alpha_stop)

        
        # M-step
        a_mat = np.divide(np.array(new_a_numerator) + smoothing_constant, np.array(np.sum(new_a_numerator, axis=0)).T + (smoothing_constant * (len(states) + 1)))
        b_mat = np.divide(np.array(new_b_numerator).T + smoothing_constant, np.array(new_b_denominator) + (smoothing_constant * (len(unique_words) + 1)))

        # Updating trans_probs and output_probs for next iteration
        trans_matrix = a_mat.T
        output_matrix = b_mat.T

        # Updating start and end trans_probs for next iteration 
        start_trans = (np.array(new_start_trans) + smoothing_constant) / (sum(new_start_trans) + (smoothing_constant * (len(states) + 1)))
        stop_trans = (np.array(new_stop_trans) + smoothing_constant) / (sum(new_stop_trans) + (smoothing_constant * (len(states) + 1)))

        # Calculate current iteration log likelihood
        log_likelihoods[iteration] = get_log_likelihood(train, start_trans, stop_trans, trans_matrix, output_matrix, states, unique_words)

        print("Current log likelihood is", log_likelihoods[iteration])

        # Calculate log likelihood change
        log_likelihood_change = abs((log_likelihoods[iteration] - log_likelihoods[iteration-1]) / log_likelihoods[iteration-1])

        # Terminate when fractional change of (log)likelihoods falls below thresh
        if log_likelihood_change < thresh:
            print("fractional change between log likelihoods below threshold")
            break
    
    # Save outputs into files

    # Save output_probs
    all_print = ''
    for i in range(len(unique_words)):
        for j, state in enumerate(states):
            prob = output_matrix[j, i]
            all_print += unique_words[i] + "\t" + state + "\t" + str(prob) + "\n"

    print(all_print, file=open(out_output_filename, "w"))

    # Save trans_probs
    all_print = ''
    for i in range(len(states)):
        for j in range(len(states)):
            prob = trans_matrix[i,j]
            all_print += states[i] + "\t" + states[j] + "\t" + str(prob) + "\n"

    start_states = [('START', j) for j in states]
    end_states = [(j, 'END') for j in states]

    for (i, j) in start_states:
        state_index = states.index(j)
        prob = start_trans[state_index]
        all_print += i + "\t" + j + "\t" + str(prob) + "\n"

    for (i, j) in end_states:
        state_index = states.index(i)
        prob = stop_trans[state_index]
        all_print += i + "\t" + j + "\t" + str(prob) + "\n"

    print(all_print, file=open(out_trans_filename, "w"))


def output_semantics(fb_output_file):
    # File format: Price    State   Probability
    with open(fb_output_file) as f:
        output = f.read()
        output = output.split('\n')
        output = list(filter(lambda x: x != '', output))
        output = [i.split("\t") for i in output]
        output = dict([*map(lambda x: ((x[0], x[1]), float(x[2])), output)])

    # Create dictionary to view semantics
    s = {}
    states = ["s0", "s1", "s2"]
    outcomes = ["x_zero","x_positive","x_negative"]
    for state in states:
        for outcome in outcomes:
            s[(state,outcome)] = 0

    for k,v in output.items():
        if k[0] != "unseen":
            if int(k[0]) == 0:
                s[(k[1],outcomes[0])] += v
            elif 0 < int(k[0]) <= 6:
                s[(k[1],outcomes[1])] += v
            elif -6 <= int(k[0]) < 0:
                s[(k[1],outcomes[2])] += v
    return s


def cat_predict(in_test_filename, in_trans_probs_filename, in_output_probs_filename, in_states_filename,
                out_predictions_file):

    # Process states
    with open(in_states_filename) as f:
        states = f.readlines()
    states = [i.strip("\n") for i in states if len(i.strip()) != 0]
    N = len(states)

    # Process output probs
    with open(in_output_probs_filename) as f:
        lines = f.readlines()
    lines = [i.strip("\n").split("\t") for i in lines if len(i.strip()) != 0]
    output_probs = np.zeros((N, 13)) #-6 to 6
    for price, state, prob in lines:
        state_index = states.index(state)
        if price != "unseen":
            output_probs[state_index, int(price)+6] = float(prob) # +6 to convert [-6, 6] to [0, 12]

    # Get highest output probs
    highest_prob_output = np.zeros(N)
    for i in range(N):
        highest_index = np.where(output_probs[i] == np.amax(output_probs[i]))[0]
        highest_prob_output[i] = highest_index

    # Process trans probs
    with open(in_trans_probs_filename) as f:
        lines = f.readlines()
    lines = [i.strip("\n").split("\t") for i in lines if len(i.strip()) != 0]
    trans_probs = np.zeros((N, N))
    initial_trans_probs = np.zeros(N)
    end_trans_probs = np.zeros(N)
    for prev, curr, prob in lines:
        if prev == 'START':
            curr_index = states.index(curr)
            initial_trans_probs[curr_index] = float(prob)
        elif curr == 'END':
            prev_index = states.index(prev)
            end_trans_probs[prev_index] = float(prob)
        else:
            prev_index = states.index(prev)
            curr_index = states.index(curr)
            trans_probs[prev_index, curr_index] = float(prob)

    # Process test data
    with open(in_test_filename) as f:
        lines = f.readlines()
    lines = [i.strip("\n") for i in lines]

    # Process all trades
    all_trades = [[]]
    index = 0
    for i in range(len(lines)):
        if i == len(lines) - 1:
            break
        elif lines[i] == '':
            all_trades.append([])
            index += 1
        else:
            all_trades[index].append(lines[i])

    opt_output = []

    for trade in all_trades:
        n = len(trade)
        output_matrix_trade = np.zeros((N, n))

        for i in range(n):
            for s in range(N):
                price = int(trade[i])
                output_matrix_trade[s, i] = output_probs[s, price+6]

        # get alpha for 1 to n-1
        alpha = forward(initial_trans_probs, trans_probs, output_matrix_trade, states, trade)
        alpha_next = np.zeros(N)

        # get alpha for n
        for i in range(N):
            temp = sum(alpha[:, -1] * trans_probs[:, i] * highest_prob_output[i])
            alpha_next[i] = temp

        # choose state that max alpha(n)
        state_max = np.argmax(alpha_next)
        opt_output.append(highest_prob_output[state_max] - 6)

    # Save predictions
    all_print = ''
    for i in opt_output:
        all_print += str(int(i)) + "\n"

    print(all_print, file=open(out_predictions_file, "w"))
    

def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct / len(predicted_tags)


def evaluate_ave_squared_error(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    error = 0.0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        error += (int(pred) - int(truth)) ** 2
    return error / len(predicted_tags), error, len(predicted_tags)


def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir = './' #your working dir

    in_train_filename = f'{ddir}/twitter_train.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')

    in_train_filename   = f'{ddir}/twitter_train_no_tag.txt'
    in_tag_filename     = f'{ddir}/twitter_tags.txt'
    out_trans_filename  = f'{ddir}/trans_probs4.txt'
    out_output_filename = f'{ddir}/output_probs4.txt'
    max_iter = 10
    seed     = 8
    thresh   = 1e-4
    forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh)

    trans_probs_filename3 =  f'{ddir}/trans_probs3.txt'
    output_probs_filename3 = f'{ddir}/output_probs3.txt'
    viterbi_predictions_filename3 = f'{ddir}/fb_predictions3.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename3, output_probs_filename3, in_test_filename,
                     viterbi_predictions_filename3)
    correct, total, acc = evaluate(viterbi_predictions_filename3, in_ans_filename)
    print(f'iter 0 prediction accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename4 =  f'{ddir}/trans_probs4.txt'
    output_probs_filename4 = f'{ddir}/output_probs4.txt'
    viterbi_predictions_filename4 = f'{ddir}/fb_predictions4.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename4, output_probs_filename4, in_test_filename,
                     viterbi_predictions_filename4)
    correct, total, acc = evaluate(viterbi_predictions_filename4, in_ans_filename)
    print(f'iter 10 prediction accuracy:   {correct}/{total} = {acc}')

    in_train_filename   = f'{ddir}/cat_price_changes_train.txt'
    in_tag_filename     = f'{ddir}/cat_states.txt'
    out_trans_filename  = f'{ddir}/cat_trans_probs.txt'
    out_output_filename = f'{ddir}/cat_output_probs.txt'
    max_iter = 1000000
    seed     = 8
    thresh   = 1e-4
    forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh)

    in_test_filename         = f'{ddir}/cat_price_changes_dev.txt'
    in_trans_probs_filename  = f'{ddir}/cat_trans_probs.txt'
    in_output_probs_filename = f'{ddir}/cat_output_probs.txt'
    in_states_filename       = f'{ddir}/cat_states.txt'
    predictions_filename     = f'{ddir}/cat_predictions.txt'
    cat_predict(in_test_filename, in_trans_probs_filename, in_output_probs_filename, in_states_filename,
                predictions_filename)

    in_ans_filename     = f'{ddir}/cat_price_changes_dev_ans.txt'
    ave_sq_err, sq_err, num_ex = evaluate_ave_squared_error(predictions_filename, in_ans_filename)
    print(f'average squared error for {num_ex} examples: {ave_sq_err}')

if __name__ == '__main__':
    run()
