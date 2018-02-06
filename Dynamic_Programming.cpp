//-------------------------------------------------------------------------------------------------------------------
// 121. Best Time to Buy and Sell Stock
// Say you have an array for which the ith element is the price of a given stock on day i.

// If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.

// Example 1:
// Input: [7, 1, 5, 3, 6, 4]
// Output: 5

// max. difference = 6-1 = 5 (not 7-1 = 6, as selling price needs to be larger than buying price)
// Example 2:
// Input: [7, 6, 4, 3, 1]
// Output: 0

// In this case, no transaction is done, i.e. max profit = 0.
//https://discuss.leetcode.com/topic/107998/most-consistent-ways-of-dealing-with-the-series-of-stock-problems/2
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        if(n == 0) return 0;
        int Tik0 = 0, Tik1 = INT_MIN;
        for(int i = 0; i < n; i++)
        {
            Tik0 = max(Tik0, Tik1 + prices[i]);
            Tik1 = max(Tik1, -prices[i]);
        }
        return Tik0;
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 122. Best Time to Buy and Sell Stock II
// Say you have an array for which the ith element is the price of a given stock on day i.

// Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times). However, you may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        if(n == 0) return 0;
        int Tik0 = 0, Tik1 = INT_MIN;
        for(int i = 0; i < n; i++)
        {
            Tik0 = max(Tik0, Tik1 + prices[i]);
            Tik1 = max(Tik1, Tik0 - prices[i]);
        }
        return Tik0;
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 123. Best Time to Buy and Sell Stock III
// Say you have an array for which the ith element is the price of a given stock on day i.

// Design an algorithm to find the maximum profit. You may complete at most two transactions.
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        if(n == 0) return 0;
        int Ti20 = 0, Ti21 = INT_MIN, Ti10 = 0, Ti11 = INT_MIN;
        for(int i = 0; i < n; i++)
        {
            Ti20 = max(Ti20, Ti21 + prices[i]);
            Ti21 = max(Ti21, Ti10 - prices[i]);
            Ti10 = max(Ti10, Ti11 + prices[i]);
            Ti11 = max(Ti11, - prices[i]);
        }
        return Ti20;
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 188. Best Time to Buy and Sell Stock IV
// Say you have an array for which the ith element is the price of a given stock on day i.

// Design an algorithm to find the maximum profit. You may complete at most k transactions.

// Note:
// You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).

//https://discuss.leetcode.com/topic/107998/most-consistent-ways-of-dealing-with-the-series-of-stock-problems
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        int n = prices.size();
        if(n == 0) return 0;
        if(k >= n)
        {
            int Tik0 = 0, Tik1 = INT_MIN;
            for(int i = 0; i < n; i++)
            {
                Tik0 = max(Tik0, Tik1 + prices[i]);
                Tik1 = max(Tik1, Tik0 - prices[i]);
            }
            return Tik0;
        }
        vector<int> Tik0(k + 1, 0);
        vector<int> Tik1(k + 1, INT_MIN);
        for(int i = 0; i < n; i++)
        {
            for(int j = k; j > 0; j--)
            {
                Tik0[j] = max(Tik0[j], Tik1[j] + prices[i]);
                Tik1[j] = max(Tik1[j], Tik0[j - 1] - prices[i]);
            }
        }
        return Tik0[k];
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 309. Best Time to Buy and Sell Stock with Cooldown
// Say you have an array for which the ith element is the price of a given stock on day i.

// Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times) with the following restrictions:

// You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
// After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)
// Example:

// prices = [1, 2, 3, 0, 2]
// maxProfit = 3
// transactions = [buy, sell, cooldown, buy, sell]
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        if(n == 0) return 0;
        int Tik0_p = 0, Tik0 = 0, Tik1 = INT_MIN;
        for(int i = 0; i < n; i++)
        {
            int Tik0_old = Tik0;
            Tik0 = max(Tik0, Tik1 + prices[i]);
            Tik1 = max(Tik1, Tik0_p - prices[i]);
            Tik0_p = Tik0_old;
        }
        return Tik0;
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 714. Best Time to Buy and Sell Stock with Transaction Fee
// Your are given an array of integers prices, for which the i-th element is the price of a given stock on day i; and a non-negative integer fee representing a transaction fee.

// You may complete as many transactions as you like, but you need to pay the transaction fee for each transaction. You may not buy more than 1 share of a stock at a time (ie. you must sell the stock share before you buy again.)

// Return the maximum profit you can make.

// Example 1:
// Input: prices = [1, 3, 2, 8, 4, 9], fee = 2
// Output: 8
// Explanation: The maximum profit can be achieved by:
// Buying at prices[0] = 1
// Selling at prices[3] = 8
// Buying at prices[4] = 4
// Selling at prices[5] = 9
// The total profit is ((8 - 1) - 2) + ((9 - 4) - 2) = 8.
class Solution {
public:
    int maxProfit(vector<int>& prices, int fee) {
        int n = prices.size();
        if(n == 0) return 0;
        long Tik0 = 0, Tik1 = INT_MIN;
        for(int i = 0; i < n; i++)
        {
            Tik0 = max(Tik0, Tik1 + prices[i] - fee);
            Tik1 = max(Tik1, Tik0 - prices[i]);
        }
        return Tik0;
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 87. Scramble String
// Given a string s1, we may represent it as a binary tree by partitioning it to two non-empty substrings recursively.

// Below is one possible representation of s1 = "great":

//     great
//    /    \
//   gr    eat
//  / \    /  \
// g   r  e   at
//            / \
//           a   t
// To scramble the string, we may choose any non-leaf node and swap its two children.

// For example, if we choose the node "gr" and swap its two children, it produces a scrambled string "rgeat".

//     rgeat
//    /    \
//   rg    eat
//  / \    /  \
// r   g  e   at
//            / \
//           a   t
// We say that "rgeat" is a scrambled string of "great".

// Similarly, if we continue to swap the children of nodes "eat" and "at", it produces a scrambled string "rgtae".

//     rgtae
//    /    \
//   rg    tae
//  / \    /  \
// r   g  ta  e
//        / \
//       t   a
// We say that "rgtae" is a scrambled string of "great".

// Given two strings s1 and s2 of the same length, determine if s2 is a scrambled string of s1.
//https://www.jiuzhang.com/solution/scramble-string/
class Solution {
public:
    bool isScramble(string s1, string s2) {
        if (s1 == s2) return true;
        int size = s1.size();
        int value1 = 0, value2 = 0;
        for (int i = 0; i < size; ++i) 
        {
            value1 += (s1[i]-'a');
            value2 += (s2[i]-'a');
        }
        if (value1 != value2) return false; 
        for (int i = 1; i < size; i++) 
        {
            if (isScramble(s1.substr(0, i), s2.substr(0, i)) && isScramble(s1.substr(i), s2.substr(i))) return true;
            if (isScramble(s1.substr(0, i), s2.substr(size - i)) && isScramble(s1.substr(i), s2.substr(0, size - i))) return true;
        }
        return false;
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 304. Range Sum Query 2D - Immutable

// Given a 2D matrix matrix, find the sum of the elements inside the rectangle defined by its upper left corner (row1, col1) and lower right corner (row2, col2).

// Range Sum Query 2D
// The above rectangle (with the red border) is defined by (row1, col1) = (2, 1) and (row2, col2) = (4, 3), which contains sum = 8.

// Example:
// Given matrix = [
//   [3, 0, 1, 4, 2],
//   [5, 6, 3, 2, 1],
//   [1, 2, 0, 1, 5],
//   [4, 1, 0, 1, 7],
//   [1, 0, 3, 0, 5]
// ]

// sumRegion(2, 1, 4, 3) -> 8
// sumRegion(1, 1, 2, 2) -> 11
// sumRegion(1, 2, 2, 4) -> 12
class NumMatrix {
public:
    NumMatrix(vector<vector<int>> matrix) {
        int m = matrix.size();
        if(m == 0) return;
        int n = matrix[0].size();
        if(n == 0) return;
        mt.resize(m + 1, vector<int>(n + 1, 0));
        for(int i = 1; i <= m; ++i)
        {
            for(int j = 1; j <= n; ++j)
            {
                mt[i][j] += matrix[i - 1][j - 1] + mt[i - 1][j] + mt[i][j - 1] - mt[i - 1][j - 1];
            }
        }
    }
 
    int sumRegion(int row1, int col1, int row2, int col2) {
        return mt[row2 + 1][col2 + 1] - mt[row2 + 1][col1] - mt[row1][col2 + 1] + mt[row1][col1];
    }
    
private:
    vector<vector<int>> mt;
};


//-------------------------------------------------------------------------------------------------------------------
// 354. Russian Doll Envelopes

// You have a number of envelopes with widths and heights given as a pair of integers (w, h). One envelope can fit into another if and only if both the width and height of one envelope is greater than the width and height of the other envelope.

// What is the maximum number of envelopes can you Russian doll? (put one inside other)

// Example:
// Given envelopes = [[5,4],[6,4],[6,7],[2,3]], the maximum number of envelopes you can Russian doll is 3 ([2,3] => [5,4] => [6,7]).
//????????????????????????? still have binary search method
//http://www.cnblogs.com/grandyang/p/5568818.html
class Solution {
public:
    int maxEnvelopes(vector<pair<int, int>>& envelopes) {
        int res = 0, n = envelopes.size();
        vector<int> dp(n, 1);
        sort(envelopes.begin(), envelopes.end());
        for (int i = 0; i < n; ++i) 
        {
            for (int j = 0; j < i; ++j) 
            {
                if (envelopes[i].first > envelopes[j].first && envelopes[i].second > envelopes[j].second) 
                {
                    dp[i] = max(dp[i], dp[j] + 1);
                }
            }
            res = max(res, dp[i]);
        }
        return res;
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 403. Frog Jump
// A frog is crossing a river. The river is divided into x units and at each unit there may or may not exist a stone. The frog can jump on a stone, but it must not jump into the water.

// Given a list of stones' positions (in units) in sorted ascending order, determine if the frog is able to cross the river by landing on the last stone. Initially, the frog is on the first stone and assume the first jump must be 1 unit.

// If the frog's last jump was k units, then its next jump must be either k - 1, k, or k + 1 units. Note that the frog can only jump in the forward direction.

// Note:

// The number of stones is ≥ 2 and is < 1,100.
// Each stone's position will be a non-negative integer < 231.
// The first stone's position is always 0.
// Example 1:

// [0,1,3,5,6,8,12,17]

// There are a total of 8 stones.
// The first stone at the 0th unit, second stone at the 1st unit,
// third stone at the 3rd unit, and so on...
// The last stone at the 17th unit.

// Return true. The frog can jump to the last stone by jumping 
// 1 unit to the 2nd stone, then 2 units to the 3rd stone, then 
// 2 units to the 4th stone, then 3 units to the 6th stone, 
// 4 units to the 7th stone, and 5 units to the 8th stone.
// Example 2:

// [0,1,2,3,4,8,9,11]

// Return false. There is no way to jump to the last stone as 
// the gap between the 5th and 6th stone is too large.
//http://www.cnblogs.com/grandyang/p/5888439.html
class Solution {
public:
    bool canCross(vector<int>& stones) {
        unordered_map<int, bool> m;
        return helper(stones, 0, 0, m);
    }
    bool helper(vector<int>& stones, int pos, int jump, unordered_map<int, bool>& m) {
        int n = stones.size(), key = pos | jump << 12;
        if (pos >= n - 1) return true;
        if (m.count(key)) return m[key];
        for (int i = pos + 1; i < n; ++i) 
        {
            int dist = stones[i] - stones[pos];
            if (dist < jump - 1) continue;
            if (dist > jump + 1) return m[key] = false;
            if (helper(stones, i, dist, m)) return m[key] = true;
        }
        return m[key] = false;
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 265. Paint House II
// There are a row of n houses, each house can be painted with one of the k colors. The cost of painting each house with a certain color is different. You have to paint all the houses such that no two adjacent houses have the same color.

// The cost of painting each house with a certain color is represented by a n x k cost matrix. For example, costs[0][0] is the cost of painting house 0 with color 0; costs[1][2] is the cost of painting house 1 with color 2, and so on... Find the minimum cost to paint all houses.

// Note:
// All costs are positive integers.
//http://www.cnblogs.com/grandyang/p/5322870.html
class Solution {
public:
    int minCostII(vector<vector<int>>& costs) {
        if (costs.empty() || costs[0].empty()) return 0;
        vector<vector<int>> dp = costs;
        int min1 = -1, min2 = -1;
        for (int i = 0; i < dp.size(); ++i) 
        {
            int last1 = min1, last2 = min2;
            min1 = -1; min2 = -1;
            for (int j = 0; j < dp[i].size(); ++j) 
            {
                if (j != last1) dp[i][j] += last1 < 0 ? 0 : dp[i - 1][last1];
                else dp[i][j] += last2 < 0 ? 0 : dp[i - 1][last2];
                if (min1 < 0 || dp[i][j] < dp[i][min1]) 
                {
                    min2 = min1; min1 = j;
                } 
                else if (min2 < 0 || dp[i][j] < dp[i][min2]) 
                {
                    min2 = j;
                }
            }
        }
        return dp.back()[min1];
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 53. Maximum Subarray
// Find the contiguous subarray within an array (containing at least one number) which has the largest sum.

// For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
// the contiguous subarray [4,-1,2,1] has the largest sum = 6.
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int n = nums.size();
        if(n == 0) return 0;
        if(n == 1) return nums[0];
        int res = nums[0], sum = nums[0];
        for(int i = 1; i < n; i++)
        {
            if(sum >= 0) sum += nums[i];
            else sum = nums[i];
            res = max(res, sum);
        }
        return res;
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 70. Climbing Stairs
// You are climbing a stair case. It takes n steps to reach to the top.

// Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

// Note: Given n will be a positive integer.


// Example 1:

// Input: 2
// Output:  2
// Explanation:  There are two ways to climb to the top.

// 1. 1 step + 1 step
// 2. 2 steps
// Example 2:

// Input: 3
// Output:  3
// Explanation:  There are three ways to climb to the top.

// 1. 1 step + 1 step + 1 step
// 2. 1 step + 2 steps
// 3. 2 steps + 1 step
class Solution {
public:
    int climbStairs(int n) {
        vector<int> dp(n + 1, 1);
        for(int i = 2; i <= n; i++)
        {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 198. House Robber
// You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

// Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.
class Solution {
public:
    int rob(vector<int>& nums) {
        if(nums.size() == 0) return 0;
        vector<int> dp(nums.size(), 0);
        dp[0] = nums[0];
        dp[1] = max(nums[0], nums[1]);
        for(int i = 2; i < nums.size(); i++)
        {
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        return dp[nums.size() - 1];
    }
};


//-------------------------------------------------------------------------------------------------------------------
//10. Regular Expression Matching
// Implement regular expression matching with support for '.' and '*'.

// '.' Matches any single character.
// '*' Matches zero or more of the preceding element.

// The matching should cover the entire input string (not partial).

// The function prototype should be:
// bool isMatch(const char *s, const char *p)

// Some examples:
// isMatch("aa","a") → false
// isMatch("aa","aa") → true
// isMatch("aaa","aa") → false
// isMatch("aa", "a*") → true
// isMatch("aa", ".*") → true
// isMatch("ab", ".*") → true
// isMatch("aab", "c*a*b") → true
class Solution {
public:
    bool isMatch(string s, string p) {
        int m = s.length(), n = p.length();
        vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
        dp[0][0] = true;      
        for(int i = 0; i <= s.length(); i++)
        {
            for(int j = 1; j <= p.length(); j++)
            {
                if(p[j - 1] != '.' && p[j - 1] != '*') 
                {
                    if(i > 0 && s[i - 1] == p[j - 1] && dp[i - 1][j - 1])
                        dp[i][j] = true;
                }
                else if(p[j - 1] == '.') 
                {
                    if(i > 0 && dp[i - 1][j - 1]) dp[i][j] = true;
                }
                else if(j > 1)
                {  
                    if(dp[i][j - 1] || dp[i][j - 2])  // match 0 or 1 preceding element
                        dp[i][j] = true;
                    else if(i > 0 && (p[j - 2] == s[i - 1] || p[j - 2] == '.') && dp[i - 1][j]) // match multiple preceding elements
                        dp[i][j] = true;
                }
            }
        }
        return dp[m][n];
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 44. Wildcard Matching
// Implement wildcard pattern matching with support for '?' and '*'.

// '?' Matches any single character.
// '*' Matches any sequence of characters (including the empty sequence).

// The matching should cover the entire input string (not partial).

// The function prototype should be:
// bool isMatch(const char *s, const char *p)

// Some examples:
// isMatch("aa","a") → false
// isMatch("aa","aa") → true
// isMatch("aaa","aa") → false
// isMatch("aa", "*") → true
// isMatch("aa", "a*") → true
// isMatch("ab", "?*") → true
// isMatch("aab", "c*a*b") → false
class Solution {
public:
    bool isMatch(string s, string p) {
        int m = s.length(), n = p.length();
        vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
        dp[0][0] = true;   
        for (int i = 1; i <= n; i++) dp[0][i] = dp[0][i - 1] && p[i - 1] == '*';
        for(int i = 1; i <= s.length(); i++)
        {
            for(int j = 1; j <= p.length(); j++)
            {
                if(p[j - 1] != '?' && p[j - 1] != '*') 
                {
                    if(i > 0 && s[i - 1] == p[j - 1] && dp[i - 1][j - 1])
                        dp[i][j] = true;
                }
                else if(p[j - 1] == '?') 
                {
                    if(i > 0 && dp[i - 1][j - 1]) dp[i][j] = true;
                }
                else
                {  
                    dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
                }
            }
        }
        return dp[m][n];
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 139. Word Break
// Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, determine if s can be segmented into a space-separated sequence of one or more dictionary words. You may assume the dictionary does not contain duplicate words.

// For example, given
// s = "leetcode",
// dict = ["leet", "code"].

// Return true because "leetcode" can be segmented as "leet code".

// UPDATE (2017/1/4):
// The wordDict parameter had been changed to a list of strings (instead of a set of strings). Please reload the code definition to get the latest changes.
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        if(s.length() == 0) return false;
        vector<bool> dp(s.length() + 1, false);
        dp[0] = true;
        for(int i = 1; i <= s.length(); i++)
        {
            for(int j = 0; j <= i; j++)
            {
                if(dp[j] && find(wordDict.begin(), wordDict.end(), s.substr(j, i - j)) != wordDict.end()) dp[i] = true;
            }
        }
        
        return dp[s.length()];
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 140. Word Break II
// Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, add spaces in s to construct a sentence where each word is a valid dictionary word. You may assume the dictionary does not contain duplicate words.

// Return all such possible sentences.

// For example, given
// s = "catsanddog",
// dict = ["cat", "cats", "and", "sand", "dog"].

// A solution is ["cats and dog", "cat sand dog"].
class Solution {
public:
    vector<string> wordBreak(string s, vector<string> &dict) {
        string result;
        vector<string> solutions;
        int len = s.size();
        vector<bool> possible(len + 1, true);
        GetAllSolution(0, s, dict, len, result, solutions, possible);
        return solutions;
    }

    void GetAllSolution(int start, const string& s, const vector<string> &dict, int len, string& result, vector<string>& solutions, vector<bool>& possible)
    {
        if(start == len)
        {
            solutions.push_back(result.substr(0, result.size() - 1));
            return;
        }
        for(int i = start; i < len; i++)
        {
            string piece = s.substr(start, i - start + 1);
            if(find(dict.begin(), dict.end(), piece) != dict.end() && possible[i + 1])
            {
                result.append(piece).append(" ");
                int beforeChange = solutions.size();
                GetAllSolution(i + 1, s, dict, len, result, solutions, possible);
                if(solutions.size() == beforeChange)
                    possible[i + 1] = false;
                result.resize(result.size() - piece.size() - 1);
            }
        }
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 72. Edit Distance
// Given two words word1 and word2, find the minimum number of steps required to convert word1 to word2. (each operation is counted as 1 step.)

// You have the following 3 operations permitted on a word:

// a) Insert a character
// b) Delete a character
// c) Replace a character
class Solution {
public:
    int minDistance(string word1, string word2) {
        int m = word1.length(), n = word2.length();
        if(m == 0 && n == 0) return 0;
        if(m == 0) return n;
        if(n == 0) return m;
        int dp[m + 1][n + 1];
        dp[0][0] = 0;
        for(int i = 1; i <= m; i++)
        {
            dp[i][0] = i;
        }
        for(int i = 1; i <= n; i++)
        {
            dp[0][i] = i;
        }
        for(int i = 1; i <= m; i++)
        {
            for(int j = 1; j <= n; j++)
            {
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1);
                if(word1[i - 1] == word2[j - 1]) dp[i][j] = min(dp[i][j], dp[i - 1][j - 1]);
                else dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + 1);
            }
        }
        
        return dp[m][n];
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 95. Unique Binary Search Trees II
// Given an integer n, generate all structurally unique BST's (binary search trees) that store values 1...n.

// For example,
// Given n = 3, your program should return all 5 unique BST's shown below.

//    1         3     3      2      1
//     \       /     /      / \      \
//      3     2     1      1   3      2
//     /     /       \                 \
//    2     1         2                 3
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<TreeNode*> generateTrees(int n) {
        if(n == 0) return {};
        return genBST(1, n);
    }
    
    vector<TreeNode*> genBST(int min, int max)
    {
        vector<TreeNode*> ret;
        if(min > max)
        {
            ret.push_back(NULL);
            return ret;
        }
        
        for(int i = min; i <= max; i++)
        {
            vector<TreeNode*> leftSub = genBST(min, i - 1);
            vector<TreeNode*> rightSub = genBST(i + 1, max);
            for(int j = 0; j < leftSub.size(); j++)
            {
                for(int k = 0; k < rightSub.size(); k++)
                {
                    TreeNode *root = new TreeNode(i);
                    root->left = leftSub[j];
                    root->right = rightSub[k];
                    ret.push_back(root);
                }
            }
        }
        return ret;
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 279. Perfect Squares
// Given a positive integer n, find the least number of perfect square numbers (for example, 1, 4, 9, 16, ...) which sum to n.

// For example, given n = 12, return 3 because 12 = 4 + 4 + 4; given n = 13, return 2 because 13 = 4 + 9.
class Solution {
public:
    int numSquares(int n) {
        vector<int> dp(n + 1, INT_MAX);
        dp[0] = 0;
        for(int i = 0; i <= n; i++)
        {
            for(int j = 1; j * j <= n - i; j++)
            {
                dp[i + j * j] = min(dp[i + j * j], dp[i] + 1);
            }
        }
        
        return dp.back();
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 312. Burst Balloons
// Given n balloons, indexed from 0 to n-1. Each balloon is painted with a number on it represented by array nums. You are asked to burst all the balloons. If the you burst balloon i you will get nums[left] * nums[i] * nums[right] coins. Here left and right are adjacent indices of i. After the burst, the left and right then becomes adjacent.

// Find the maximum coins you can collect by bursting the balloons wisely.

// Note: 
// (1) You may imagine nums[-1] = nums[n] = 1. They are not real therefore you can not burst them.
// (2) 0 ≤ n ≤ 500, 0 ≤ nums[i] ≤ 100

// Example:

// Given [3, 1, 5, 8]

// Return 167

//     nums = [3,1,5,8] --> [3,5,8] -->   [3,8]   -->  [8]  --> []
//    coins =  3*1*5      +  3*5*8    +  1*3*8      + 1*8*1   = 167
//https://www.youtube.com/watch?v=IFNibRVgFBo
class Solution {
public:
    int maxCoins(vector<int>& nums) {
        int n = nums.size();
        nums.insert(nums.begin(), 1);
        nums.push_back(1);
        vector<vector<int> > dp(nums.size(), vector<int>(nums.size() , 0));
        for (int len = 1; len <= n; ++len) 
        {
            for (int left = 1; left <= n - len + 1; ++left) 
            {
                int right = left + len - 1;
                for (int k = left; k <= right; ++k) 
                {
                    dp[left][right] = max(dp[left][right], nums[left - 1] * nums[k] * nums[right + 1] + dp[left][k - 1] + dp[k + 1][right]);
                }
            }
        }
        
        return dp[1][n];
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 115. Distinct Subsequences
// Given a string S and a string T, count the number of distinct subsequences of S which equals T.

// A subsequence of a string is a new string which is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (ie, "ACE" is a subsequence of "ABCDE" while "AEC" is not).

// Here is an example:
// S = "rabbbit", T = "rabbit"

// Return 3.
// Ø r a b b b i t
// Ø 1 1 1 1 1 1 1 1
// r 0 1 1 1 1 1 1 1
// a 0 0 1 1 1 1 1 1
// b 0 0 0 1 2 3 3 3
// b 0 0 0 0 1 3 3 3
// i 0 0 0 0 0 0 3 3
// t 0 0 0 0 0 0 0 3
//http://www.cnblogs.com/grandyang/p/4294105.html
class Solution {
public:
    int numDistinct(string S, string T) {
        int dp[T.size() + 1][S.size() + 1];
        for (int i = 0; i <= S.size(); ++i) dp[0][i] = 1;    
        for (int i = 1; i <= T.size(); ++i) dp[i][0] = 0;    
        for (int i = 1; i <= T.size(); ++i) 
        {
            for (int j = 1; j <= S.size(); ++j) 
            {
                dp[i][j] = dp[i][j - 1] + (T[i - 1] == S[j - 1] ? dp[i - 1][j - 1] : 0);
            }
        }
        return dp[T.size()][S.size()];
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 32. Longest Valid Parentheses
// Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.

// For "(()", the longest valid parentheses substring is "()", which has length = 2.

// Another example is ")()())", where the longest valid parentheses substring is "()()", which has length = 4.
class Solution {
public:
    int longestValidParentheses(string s) {
        int idx = -1, n = s.length(), len = 0, res = 0;
        stack<int> h;
        for(int i = 0; i < n; i++)
        {
            if(s[i] == '(') h.push(i);
            else
            {
                if(!h.empty())
                {
                    h.pop();
                    if(h.empty()) res = max(res, i - idx);
                    else res = max(res, i - h.top());
                }
                else idx = i;
            }
        }
        return res;
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 647. Palindromic Substrings
// Given a string, your task is to count how many palindromic substrings in this string.

// The substrings with different start indexes or end indexes are counted as different substrings even they consist of same characters.

// Example 1:
// Input: "abc"
// Output: 3
// Explanation: Three palindromic strings: "a", "b", "c".
// Example 2:
// Input: "aaa"
// Output: 6
// Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
class Solution {
public:
    int countSubstrings(string s) {
        int n = s.size(), res = 0;
        vector<vector<bool>> dp(n, vector<bool>(n, false));
        for (int i = n - 1; i >= 0; --i) 
        {
            for (int j = i; j < n; ++j) 
            {
                dp[i][j] = (s[i] == s[j]) && (j - i <= 2 || dp[i + 1][j - 1]);
                if (dp[i][j]) ++res;
            }
        }
        return res;
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 322. Coin Change
// You are given coins of different denominations and a total amount of money amount. Write a function to compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

// Example 1:
// coins = [1, 2, 5], amount = 11
// return 3 (11 = 5 + 5 + 1)

// Example 2:
// coins = [2], amount = 3
// return -1.

// Note:
// You may assume that you have an infinite number of each kind of coin.
//http://www.cnblogs.com/grandyang/p/5138186.html
// Non-recursion
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        vector<int> dp(amount + 1, amount + 1);
        dp[0] = 0;
        for (int i = 1; i <= amount; ++i) 
        {
            for (int j = 0; j < coins.size(); ++j) 
            {
                if (coins[j] <= i) 
                {
                    dp[i] = min(dp[i], dp[i - coins[j]] + 1);
                }
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 97. Interleaving String
// Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and s2.

// For example,
// Given:
// s1 = "aabcc",
// s2 = "dbbca",

// When s3 = "aadbbcbcac", return true.
// When s3 = "aadbbbaccc", return false.
class Solution {
public:
    bool isInterleave(string s1, string s2, string s3) {
        int len1 = s1.length(), len2 = s2.length(), len3 = s3.length();
        if(len1 == 0) return s2 == s3;
        if(len2 == 0) return s1 == s3;
        
        vector<vector<bool>> dp(len1 + 1, vector<bool>(len2 + 1, false));
        if(len3 != 0)
        {
            if(s1[0] == s3[0]) dp[1][0] = true;
            if(s2[0] == s3[0]) dp[0][1] = true;
        }
        else return false;
        
        if(len1 + len2 != len3) return false;
        for(int i = 0; i <= len1; i++)
        {
            for(int j = 0; j <= len2; j++)
            {
                int t = i + j - 1;
                if(i > 0)
                {
                    if(dp[i - 1][j] && s3[t] == s1[i - 1]) dp[i][j] = true;
                }
                if(j > 0)
                {
                    if(dp[i][j - 1] && s3[t] == s2[j - 1]) dp[i][j] = true; 
                }
            }
        }
        if(dp[len1][len2]) return true;
        else return false;
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 174. Dungeon Game
// The demons had captured the princess (P) and imprisoned her in the bottom-right corner of a dungeon. The dungeon consists of M x N rooms laid out in a 2D grid. Our valiant knight (K) was initially positioned in the top-left room and must fight his way through the dungeon to rescue the princess.

// The knight has an initial health point represented by a positive integer. If at any point his health point drops to 0 or below, he dies immediately.

// Some of the rooms are guarded by demons, so the knight loses health (negative integers) upon entering these rooms; other rooms are either empty (0's) or contain magic orbs that increase the knight's health (positive integers).

// In order to reach the princess as quickly as possible, the knight decides to move only rightward or downward in each step.


// Write a function to determine the knight's minimum initial health so that he is able to rescue the princess.

// For example, given the dungeon below, the initial health of the knight must be at least 7 if he follows the optimal path RIGHT-> RIGHT -> DOWN -> DOWN.
class Solution {
public:
    int calculateMinimumHP(vector<vector<int>>& dungeon) {
        int m = dungeon.size();
        int n = dungeon[0].size();
        dungeon[m - 1][n - 1] = max(-dungeon[m - 1][n - 1], 0);
        for(int i = m - 2; i >= 0; --i)
        {
            dungeon[i][n - 1] = max(dungeon[i + 1][n - 1] - dungeon[i][n - 1], 0);
        }
        for(int i = n - 2; i >= 0; --i)
        {
            dungeon[m - 1][i] = max(dungeon[m - 1][i + 1] - dungeon[m - 1][i], 0);
        }
        for(int i = m - 2; i >= 0; --i)
        {
            for(int j = n - 2; j >= 0; --j)
            {
                dungeon[i][j] = max(min(dungeon[i + 1][j], dungeon[i][j + 1]) - dungeon[i][j], 0);
            }
        }
        
        return dungeon[0][0];
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 516. Longest Palindromic Subsequence
// Given a string s, find the longest palindromic subsequence's length in s. You may assume that the maximum length of s is 1000.

// Example 1:
// Input:

// "bbbab"
// Output:
// 4
// One possible longest palindromic subsequence is "bbbb".
// Example 2:
// Input:

// "cbbd"
// Output:
// 2
// One possible longest palindromic subsequence is "bb".
class Solution {
public:
    int longestPalindromeSubseq(string s) {
        int n = s.length();
        if(n == 0) return 0;
        vector<vector<int>> dp(n, vector<int>(n, 0));
        for(int i = n - 1; i >= 0; --i)
        {
            dp[i][i] = 1;
            for(int j = i + 1; j < n; ++j)
            {
                if (s[i] == s[j])
                {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } 
                else 
                {
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][n - 1];
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 132. Palindrome Partitioning II
// Given a string s, partition s such that every substring of the partition is a palindrome.

// Return the minimum cuts needed for a palindrome partitioning of s.

// For example, given s = "aab",
// Return 1 since the palindrome partitioning ["aa","b"] could be produced using 1 cut.
class Solution {
public:
    int minCut(string s) {
        int n = s.size();
        if(n <= 1) return 0;
        vector<vector<bool>> isPal(n, vector<bool>(n, false));
        for(int i = n - 1; i >= 0; i--)
        {
            for(int j = i; j < n; j++)
            {
                if((i + 1 >= j - 1 || isPal[i + 1][j - 1]) && s[i] == s[j])
                    isPal[i][j] = true;
            }
        }
        vector<int> dp(n + 1, INT_MAX);
        dp[0] = -1;
        for(int i = 1; i <= n; i++) 
        {
            for(int j = i - 1; j >= 0; j--) {
                if(isPal[j][i - 1]) 
                {
                    dp[i] = min(dp[i], dp[j] + 1);
                }
            }
        }
        
        return dp[n];
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 363. Max Sum of Rectangle No Larger Than K
// Given a non-empty 2D matrix matrix and an integer k, find the max sum of a rectangle in the matrix such that its sum is no larger than k.

// Example:
// Given matrix = [
//   [1,  0, 1],
//   [0, -2, 3]
// ]
// k = 2
// The answer is 2. Because the sum of rectangle [[0, 1], [-2, 3]] is 2 and 2 is the max number no larger than k (k = 2).
//https://www.youtube.com/watch?v=yCQN096CwWM
class Solution {
public:
    int maxSumSubmatrix(vector<vector<int>>& matrix, int k) {
        if (matrix.empty() || matrix[0].empty()) return 0;
        int m = matrix.size(), n = matrix[0].size(), res = INT_MIN;
        for (int i = 0; i < n; ++i) 
        {
            vector<int> sum(m, 0);
            for (int j = i; j < n; ++j)
            {
                for (int k = 0; k < m; ++k)
                {
                    sum[k] += matrix[k][j];
                }
                int curSum = 0, curMax = INT_MIN;
                set<int> s;
                s.insert(0);
                for (auto a : sum) 
                {
                    curSum += a;
                    auto it = s.lower_bound(curSum - k);
                    if (it != s.end()) curMax = max(curMax, curSum - *it);
                    s.insert(curSum);
                }
                res = max(res, curMax);
            }
        }
        return res;
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 321. Create Maximum Number
// Given two arrays of length m and n with digits 0-9 representing two numbers. Create the maximum number of length k <= m + n from digits of the two. The relative order of the digits from the same array must be preserved. Return an array of the k digits. You should try to optimize your time and space complexity.

// Example 1:
// nums1 = [3, 4, 6, 5]
// nums2 = [9, 1, 2, 5, 8, 3]
// k = 5
// return [9, 8, 6, 5, 3]

// Example 2:
// nums1 = [6, 7]
// nums2 = [6, 0, 4]
// k = 5
// return [6, 7, 6, 0, 4]

// Example 3:
// nums1 = [3, 9]
// nums2 = [8, 9]
// k = 3
// return [9, 8, 9]
class Solution {
public:
    vector<int> maxNumber(vector<int>& nums1, vector<int>& nums2, int k) {
        int m = nums1.size(), n = nums2.size();
        vector<int> res;
        for (int i = max(0, k - n); i <= min(k, m); ++i) {
            res = max(res, mergeVector(maxVector(nums1, i), maxVector(nums2, k - i)));
        }
        return res;
    }
    vector<int> maxVector(vector<int> nums, int k) {
        int drop = nums.size() - k;
        vector<int> res;
        for (int num : nums) {
            while (drop && res.size() && res.back() < num) {
                res.pop_back();
                --drop;
            }
            res.push_back(num);
        }
        res.resize(k);
        return res;
    }
    vector<int> mergeVector(vector<int> nums1, vector<int> nums2) {
        vector<int> res;
        while (nums1.size() + nums2.size()) {
            vector<int> &tmp = nums1 > nums2 ? nums1 : nums2;
            res.push_back(tmp[0]);
            tmp.erase(tmp.begin());
        }
        return res;
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 376. Wiggle Subsequence
// A sequence of numbers is called a wiggle sequence if the differences between successive numbers strictly alternate between positive and negative. The first difference (if one exists) may be either positive or negative. A sequence with fewer than two elements is trivially a wiggle sequence.

// For example, [1,7,4,9,2,5] is a wiggle sequence because the differences (6,-3,5,-7,3) are alternately positive and negative. In contrast, [1,4,7,2,5] and [1,7,4,5,5] are not wiggle sequences, the first because its first two differences are positive and the second because its last difference is zero.

// Given a sequence of integers, return the length of the longest subsequence that is a wiggle sequence. A subsequence is obtained by deleting some number of elements (eventually, also zero) from the original sequence, leaving the remaining elements in their original order.

// Examples:
// Input: [1,7,4,9,2,5]
// Output: 6
// The entire sequence is a wiggle sequence.

// Input: [1,17,5,10,13,15,10,5,16,8]
// Output: 7
// There are several subsequences that achieve this length. One is [1,17,10,13,10,16,8].

// Input: [1,2,3,4,5,6,7,8,9]
// Output: 2
//http://www.cnblogs.com/grandyang/p/5697621.html
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        int p = 1, q = 1, n = nums.size();
        for (int i = 1; i < n; ++i) {
            if (nums[i] > nums[i - 1]) p = q + 1;
            else if (nums[i] < nums[i - 1]) q = p + 1;
        }
        return min(n, max(p, q));
    }
};


//-------------------------------------------------------------------------------------------------------------------
// 730. Count Different Palindromic Subsequences
// Given a string S, find the number of different non-empty palindromic subsequences in S, and return that number modulo 10^9 + 7.

// A subsequence of a string S is obtained by deleting 0 or more characters from S.

// A sequence is palindromic if it is equal to the sequence reversed.

// Two sequences A_1, A_2, ... and B_1, B_2, ... are different if there is some i for which A_i != B_i.

// Example 1:
// Input: 
// S = 'bccb'
// Output: 6
// Explanation: 
// The 6 different non-empty palindromic subsequences are 'b', 'c', 'bb', 'cc', 'bcb', 'bccb'.
// Note that 'bcb' is counted only once, even though it occurs twice.
// Example 2:
// Input: 
// S = 'abcdabcdabcdabcdabcdabcdabcdabcddcbadcbadcbadcbadcbadcbadcbadcba'
// Output: 104860361
// Explanation: 
// There are 3104860382 different non-empty palindromic subsequences, which is 104860361 modulo 10^9 + 7.
//http://zxi.mytechroad.com/blog/dynamic-programming/leetcode-730-count-different-palindromic-subsequences/11
class Solution {
public:
    int countPalindromicSubsequences(const string& S) {
        int n = S.length();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        for (int i = 0; i < n; ++i) dp[i][i] = 1;
        
        for (int len = 1; len <= n; ++len) 
        {
            for (int i = 0; i < n - len; ++i) 
            {
                const int j = i + len;                
                if (S[i] == S[j]) 
                {
                    dp[i][j] = dp[i + 1][j - 1] * 2;                        
                    int l = i + 1;
                    int r = j - 1;
                    while (l <= r && S[l] != S[i]) ++l;
                    while (l <= r && S[r] != S[i]) --r;                    
                    if (l == r) dp[i][j] += 1;
                    else if (l > r) dp[i][j] += 2;
                    else dp[i][j] -= dp[l + 1][r - 1];
                } 
                else 
                {
                    dp[i][j] = dp[i][j - 1] + dp[i + 1][j] - dp[i + 1][j - 1]; 
                }
                
                dp[i][j] = (dp[i][j] + kMod) % kMod;
            }
        }
        
        return dp[0][n - 1];
    }
private:
    static constexpr long kMod = 1000000007;    
};