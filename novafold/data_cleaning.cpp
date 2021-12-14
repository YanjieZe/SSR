/**
 * @file data_cleaning.cpp
 * @brief Wash up data to get rid of invalid data and data with pseudoknots
 * @version 0.1
 * @date 2021-12-14
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include <bits/stdc++.h>
using namespace std;

pair<vector<size_t>, vector<char>> readBPSEQ(const string&);
bool pseudoknotExists(const vector<size_t>&, const vector<char>&);
void checkLists(const vector<string>&);

int main() {
    checkLists({
        "data/archiveII.lst",
        "data/bpRNAnew.nr500.canonicals.lst",
        "data/RNAStrAlign600-train.lst",
        "data/TestSetA.lst",
        "data/TestSetB.lst",
        "data/TR0-canonicals.lst",
        "data/TrainSetA.lst",
        "data/TrainSetB.lst",
        "data/TS0-canonicals.lst",
        "data/VL0-canonicals.lst",
    });

    return 0;
}

// Reference: X. Chen, Y. Li, R. Umarov et al. RNA Secondary Structure Prediction By Learning Unrolled Algorithms
namespace RNA_GLOBAL_SETTING {
    // It seems that no constraint on valid RNA characters?
    // const unordered_set<char> VALID_NT = {'A', 'a', 'G', 'g', 'C', 'c', 'U', 'u'};

    const size_t LOOP_MINDIST = 4;

    const set<pair<char, char>> VALID_PAIR = {
        {'A', 'U'}, {'U', 'A'},
        {'C', 'G'}, {'G', 'C'},
        {'G', 'U'}, {'U', 'G'}
    };
};

/**
 * @brief Read a BPSEQ file and parse its pairing information.
 * 
 * @param path string, the path of the BPSEQ file
 * @return pair<vector<size_t>, vector<char>> : {pair, seq}
 *              pair: Pairing information of a certain sequence.
 *                  pair[i] == j iff i-th nt pairs with j-th nt
 *                  pair[i] == 0 iff i-th nt is left alone
 *                  !! pair[0] is meaningless
 *              seq: Sequence string
 *                  !! seq[0] is meaningless
 */
pair<vector<size_t>, vector<char>> readBPSEQ(const string& path) {
    ifstream in(path);
    vector<size_t> pair;
    vector<char> seq;
    size_t idx, pos;
    char nt;
    
    pair.reserve(300);
    seq.reserve(300);
    pair.push_back('X');
    seq.push_back('X');

    if (!in.good())
        throw runtime_error("Unable to open file: " + path);

    while (in >> idx >> nt >> pos) {
        if (idx != pair.size())
            throw runtime_error("One nt is absent: " + to_string(idx) + " " + to_string(pos));
        if (pos < 0)
            throw runtime_error("Invalid pair: pos < 0: " + to_string(idx) + " " + to_string(pos));
        seq.push_back(toupper(nt));
        pair.push_back(pos);
        if (pos != 0 && pos <= idx) {
            if (idx - pos < RNA_GLOBAL_SETTING::LOOP_MINDIST)
                throw runtime_error("Sharp loop detected: " + to_string(idx) + " " + to_string(pos) + " " + to_string(idx - pos));
            if (pair[pos] != idx)
                throw runtime_error("pair[pos] != idx, maybe not a matching: " + to_string(idx) + " " + to_string(pos));
            if (RNA_GLOBAL_SETTING::VALID_PAIR.find({toupper(nt), seq[pos]}) == RNA_GLOBAL_SETTING::VALID_PAIR.end())
                throw runtime_error("Invalid nt combination detected: " + to_string(idx) + " " + to_string(pos) + " " + seq[idx] + '-' + seq[pos]);
        }
    }

    in.close();
    return {pair, seq};
}

/**
 * @brief check the validity sequences with pseudoknot
 * 
 * @param pair vector<size_t> Pairing information of a certain sequence.
 *             pair[i] == j iff i-th nt pairs with j-th nt
 *             pair[i] == 0 iff i-th nt is left alone
 *             !! pair[0] is meaningless
 * @param seq vector<char> Characters of the sequence, seq.size() == pair.size()
 *            !! seq[0] is meaningless
 * @exception runtime_error The input sequence is invalid.
 */
void validityCheck(const vector<size_t>& pair, const vector<char>& seq) {
    for (size_t i = 1; i < pair.size(); ++i) {
        if (pair[i]) {
            if (pair[i] < 0)
                throw runtime_error("Invalid pair: pair[i] < 0: " + to_string(i) + " " + to_string(pair[i]));
            if (pair[i] != 0 && pair[i] <= i) {
                if (i - pair[i] < RNA_GLOBAL_SETTING::LOOP_MINDIST)
                    throw runtime_error("Sharp loop detected: " + to_string(i) + " " + to_string(pair[i]) + " " + to_string(i - pair[i]));
                if (pair[pair[i]] != i)
                    throw runtime_error("pair[pair[i]] != i, maybe not a matching: " + to_string(i) + " " + to_string(pair[i]));
                if (RNA_GLOBAL_SETTING::VALID_PAIR.find({seq[i], seq[pair[i]]}) == RNA_GLOBAL_SETTING::VALID_PAIR.end())
                    throw runtime_error("Invalid nt combination detected: " + to_string(i) + " " + to_string(pair[i]) + " " + seq[i] + '-' + seq[pair[i]]);
            }
        }
    }
}

/**
 * @brief Predicate of whether pseudoknot exists in the input sequence
 *        and check the validity of it
 * 
 * @param pair vector<size_t> Pairing information of a certain sequence.
 *             pair[i] == j iff i-th nt pairs with j-th nt
 *             pair[i] == 0 iff i-th nt is left alone
 *             !! pair[0] is meaningless
 * @param seq vector<char> Characters of the sequence, seq.size() == pair.size()
 *            !! seq[0] is meaningless
 * @return true pseudoknot exists
 * @return false pseudoknot doesn't exist
 * @exception runtime_error The input sequence is invalid.
 */
bool pseudoknotExists(const vector<size_t>& pair, const vector<char>& seq) {
    stack<size_t> s;
    for (size_t i = 1; i < pair.size(); ++i) {
        if (pair[i]) {
            if (s.empty())  // i, pair[i]
                s.push(i);
            else {
                if (i < pair[s.top()]) {
                    if (pair[i] < pair[s.top()])    // s.top(), i, pair[i], pair[s.top]
                        s.push(i);
                    else if (pair[i] == pair[s.top()]) {
                        throw runtime_error("pair[i] == pair[s.top()");
                    } else {
                        validityCheck(pair, seq);
                        cout << s.top() << '\t' << i << '\t';
                        return true;    // s.top(), i, pair[s.top], pair[i],
                    }
                } else if (i == pair[s.top()]) {
                    if (pair[i] != s.top() || RNA_GLOBAL_SETTING::VALID_PAIR.find({seq[i], seq[s.top()]}) == RNA_GLOBAL_SETTING::VALID_PAIR.end())
                        throw runtime_error("pair[i] != s.top() || RNA_GLOBAL_SETTING::VALID_PAIR.find({seq[i], seq[s.top()]}) == RNA_GLOBAL_SETTING::VALID_PAIR.end()");
                    s.pop();
                } else {
                    throw runtime_error("i > pair[s.top()]");
                }
            }
        }
    }
    if (!s.empty())
        throw runtime_error("!s.empty()");
    return false;
}

/**
 * @brief Read some BPSEQ lists and check if pseudoknot exists in each of their entries.
 *        Then divide these data into train/val datasets with regard to validity and pseudoknot.
 * 
 * @param listPaths vector<string>, the paths of the list files to be checked
 */
void checkLists(const vector<string>& listPaths) {
    size_t total_cnt = 0, valid_cnt = 0, pn_cnt = 0, invalid_cnt = 0;
    ofstream out_valid_train("valid_train.lst", ofstream::out | ofstream::trunc);
    ofstream out_valid_test("valid_test.lst", ofstream::out | ofstream::trunc);
    ofstream out_pseudoknot_train("pseudoknot_train.lst", ofstream::out | ofstream::trunc);
    ofstream out_pseudoknot_test("pseudoknot_test.lst", ofstream::out | ofstream::trunc);
    ofstream out_invalid("invalid.lst", ofstream::out | ofstream::trunc);

    if (!out_valid_train.good())
        throw runtime_error("Unable to create output file: valid_train.lst" );
    if (!out_valid_test.good())
        throw runtime_error("Unable to create output file: valid_test.lst" );
    if (!out_pseudoknot_train.good())
        throw runtime_error("Unable to create output file: pseudoknot_train.lst" );
    if (!out_pseudoknot_test.good())
        throw runtime_error("Unable to create output file: pseudoknot_test.lst" );
    if (!out_invalid.good())
        throw runtime_error("Unable to create output file: invalid.lst" );

    for (const auto& listPath : listPaths) {
        vector<string> valids, pseudoknots;
        ifstream in(listPath);
        string path;

        if (!in.good()) {
            throw runtime_error("Unable to open input file: " + listPath);
        }
        cout << "-----Checking " << listPath << "-----" << endl;

        while (getline(in, path)) {
            try {
                ++total_cnt;
                auto r = readBPSEQ(path);
                if (pseudoknotExists(r.first, r.second)) {
                    cout << path << endl;
                    ++pn_cnt;
                    pseudoknots.emplace_back(move(path));
                } else {
                    ++valid_cnt;
                    valids.emplace_back(move(path));
                }
            } catch(const exception& e) {
                cout << path << ' ' << e.what() << endl;
                ++invalid_cnt;
                out_invalid << path << endl;
            }
        }

        random_shuffle(valids.begin(), valids.end());
        random_shuffle(pseudoknots.begin(), pseudoknots.end());

        size_t valids_partition = valids.size()*39/40, pseudoknots_partition = pseudoknots.size()*4/5;

        for (size_t i = 0; i < valids_partition; ++i) {
            out_valid_train << valids[i] << endl;
        }
        for (size_t i = valids_partition; i < valids.size(); ++i) {
            out_valid_test << valids[i] << endl;
        }
        for (size_t i = 0; i < pseudoknots_partition; ++i) {
            out_pseudoknot_train << pseudoknots[i] << endl;
        }
        for (size_t i = pseudoknots_partition; i < pseudoknots.size(); ++i) {
            out_pseudoknot_test << pseudoknots[i] << endl;
        }

        in.close();
    }
    out_valid_train.close();
    out_valid_test.close();
    out_pseudoknot_train.close();
    out_pseudoknot_test.close();
    out_invalid.close();

    clog << total_cnt << " sequences checked in total, " << valid_cnt << " valid, " << pn_cnt << " pseudoknots, " << invalid_cnt << " invalid" << endl;
}
