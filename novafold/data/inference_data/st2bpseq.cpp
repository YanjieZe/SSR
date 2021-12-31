#include <unistd.h>  
#include <dirent.h>  
#include <bits/stdc++.h>
using namespace std;

vector<int> getPairing(std::string& str);
vector<string> getFiles(string cate_dir);

int main() {
    string seq, pairStr;
    auto files = getFiles("st_file");
    ofstream out_lst("inference.lst");
    for (const auto& s : files) {
        cout << "Processing " << s << endl; 
        ifstream in("st_file/" + s);
        ofstream out(s.substr(0, s.size()-2) + "bpseq");
        in >> seq >> pairStr;
        auto pairs = getPairing(pairStr);
        for (size_t i = 1; i <= seq.size(); ++i) {
            out << i << ' ' << seq[i-1] << ' ' << pairs[i] << endl;
        }
        in.close();
        out.close();
        out_lst << "data/test_seq/" + s.substr(0, s.size()-2) + "bpseq" << endl;
    }
    out_lst.close();
    return 0;
}

vector<int> getPairing(std::string& str) {
    std::vector<int> pairs(str.size()+1);
    pairs[0] = 0;
    
    std::stack<size_t> s_round, s_square, s_curly;
    for (size_t i = 0; i < str.size(); ++i) {
        switch (str[i]) {
            case '.':
                pairs[i+1] = 0;
                break;
            case '(':
                s_round.push(i+1);
                break;
            case ')':
            {
                auto l = s_round.top();
                s_round.pop();
                pairs[i+1] = l;
                pairs[l] = i+1;
            }
            case '[':
                s_square.push(i+1);
                break;
            case ']':
            {
                auto l = s_square.top();
                s_square.pop();
                pairs[i+1] = l;
                pairs[l] = i+1;
            }
            case '{':
                s_curly.push(i+1);
                break;
            case '}':
            {
                auto l = s_curly.top();
                s_curly.pop();
                pairs[i+1] = l;
                pairs[l] = i+1;
            }
                break;
            default:
                throw std::runtime_error(std::string("Unexpected pairing sign ") + str[i]);
        }
    }

    return pairs;
}

vector<string> getFiles(string cate_dir) {  
    vector<string> files;
    DIR *dir;  
    struct dirent *ptr;  
   
    if ((dir=opendir(cate_dir.c_str())) == NULL) {  
        perror("Open dir error...");  
        exit(1);  
    }  
   
    while ((ptr = readdir(dir)) != NULL) {  
        if (strcmp(ptr->d_name,".") == 0 || strcmp(ptr->d_name,"..") == 0)
            continue;  
        else if (ptr->d_type == 8)
            files.push_back(ptr->d_name);  
        else if (ptr->d_type == 10)
            continue;  
        else if (ptr->d_type == 4) {  
            files.push_back(ptr->d_name);  
        }  
    }  
    closedir(dir);  
  
    sort(files.begin(), files.end());  
    return files;  
}
