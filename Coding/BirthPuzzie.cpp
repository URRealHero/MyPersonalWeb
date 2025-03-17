// Maximum is 3111
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

using namespace std;

const int MAX_INT = 3111;

// 线性筛
vector<bool> st(MAX_INT + 1, false); // True -> composite
vector<int> primes; // prime array

void get_Primes(int n){
    for(int i = 2 ; i <= n ; i++){
        if (!st[i])
            primes.push_back(i);
        for (size_t j = 0; j < primes.size() && i * primes[j] <= n; j++){
            st[i * primes[j]] = true;
            if (i % primes[j] == 0)
                break;
        }
    }
}

bool is_prime(int x){
    if (x < 2)
        return false;
    if (x < MAX_INT+1)
        return !st[x];
    for (int i = 2; i <= sqrt(x); i++){
        if (x % i == 0)
            return false;
    }
    return true;

}

// Helper for Puzzle
int num_digits(int a){
    int n = 0;
    while(a){
        a /= 10;
        n++;
    }
    return n;
}

int cat(int x, int y){
    int digits = num_digits(y);
    int power = 1;
    while (digits--) {
        power *= 10;
    }
    return x * power + y;
}

int reverse(int x){
    int res = 0;
    while(x){
        res = res * 10 + x % 10;
        x /= 10;
    }
    return res;
}

int sumd(int x){
    int res = 0;
    while(x){
        res += x % 10;
        x /= 10;
    }
    return res;
}


int birthsum(int month, int day){
    return sumd(month) + sumd(day);
}

int main(){
    get_Primes(MAX_INT);

    int p_month[] = {2, 3, 5, 7, 11};
    int daypmonth[] = {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int len_p_month = sizeof(p_month) / sizeof(p_month[0]);
    bool sol = false;

    // loop Prof's wife's birthday
    for (int i = 0; i < len_p_month; i++){
        int W_month = p_month[i];
        for (int d = 2 ; d <= daypmonth[W_month-1]; d++){
            int W_day = d;
            // m and d are primes
            if (!is_prime(d)){
                continue;
            }
            // Cond 3
            int W_sum = birthsum(W_month, d);
            if (!is_prime(W_sum)){
                continue;
            }

            // Cond 4
            int W_cat = cat(W_month, d);
            if (!is_prime(W_cat)){
                continue;
            }
            int reverse_W_cat = reverse(W_cat);
            if (is_prime(reverse_W_cat)){
                continue;
            }

            // Now Let's loop Prof's birthday
            for (int j = 0 ; j < len_p_month; j++){
                int P_month = p_month[j];
                for (int p_d = 2 ; p_d <= daypmonth[P_month-1]; p_d++){
                    if (!is_prime(p_d)){
                        continue;
                    }

                    int P_sum = birthsum(P_month, p_d);
                    if (!is_prime(P_sum)){
                        continue;
                    }
                    // Cond 5
                    int P_cat = cat(P_month, p_d);
                    if (!is_prime(P_cat)){
                        continue;
                    }

                    int reverse_P_cat = reverse(P_cat);
                    if (is_prime(reverse_P_cat)){
                        continue;
                    }

                    // Cond 6
                    int p_W_cat = cat(p_d, W_month);
                    if (!is_prime(p_W_cat)){
                        continue;
                    }

                    int rv_p_W_cat = reverse(p_W_cat);
                    if (is_prime(rv_p_W_cat)){
                        continue;
                    }

                    // Cond 7
                    int P_w_cat = cat(P_month, W_day);
                    if (!is_prime(P_w_cat)){
                        continue;
                    }

                    int rv_P_w_cat = reverse(P_w_cat);
                    if (is_prime(rv_P_w_cat)){
                        continue;
                    }

                    // Cond 8
                    int w_P_cat = cat(W_day, P_month);
                    if (is_prime(w_P_cat)){
                        continue;
                    }

                    int rv_w_P_cat = reverse(w_P_cat);
                    if (!is_prime(rv_w_P_cat)){
                        continue;
                    }

                    int W_p_cat = cat(W_month, p_d);
                    if (is_prime(W_p_cat)){
                        continue;
                    }

                    int rv_W_p_cat = reverse(W_p_cat);
                    if (is_prime(rv_W_p_cat)){
                        continue;
                    }

                    // Last Condition, idk what "sum of digits" really mean, assume summing up all digits
                    int dif = abs(W_sum - P_sum);
                    if (!is_prime(dif)){
                        continue;
                    }

                    cout << "Prof's wife's birthday: " << W_month << " " << W_day << endl;
                    cout << "Prof's birthday: " << P_month << " " << p_d << endl;
                    sol = true;

                }
            }

        }

    }
    if (!sol){
        cout << "No solution" << endl;
    }
    else
        cout << "Congrats, solution found!" << endl;
}