## TOP 143

1. **两数之和**：unordered_map遍历[LC](https://leetcode-cn.com/problems/two-sum/)
2. **两数相加**：注意最后的进位不为0，[LC](https://leetcode-cn.com/problems/add-two-numbers/)
3. **无重复字符最长子串**：滑动窗口+unordered_map记录，[LC](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)
4. **寻找两个正序数组的中位数**：相当于找两个有序数组的第k大的数，二分，先判断k/2和小数组长度大小。[LC](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)
5. **最长回文子串**: 二维动态规划
   
- dp[i] [j] = dp[i + 1] [j - 1]. if s[i + 1] [j - 1] 注意这个循环的遍历，外层是len from 0 to n，内层是i from 0 to n，j = i + len
  
6. **整数反转**：注意是否越界，可以用INT_MAX / 10或者INT_MIN / 10判断一下. [LC](https://leetcode-cn.com/problems/reverse-integer/submissions/)

7. **字符串转整数atoi**：主要就是越界的处理，和6一样，注意一点的就是-12 % 10 = -2，[LC](https://leetcode-cn.com/problems/string-to-integer-atoi/)

8. **正则表达式匹配**：二维动态规划或者递归求解。[LC](https://leetcode-cn.com/problems/string-to-integer-atoi/)

   - dp[i] [j]：s的前i个和p的前j个是否匹配，dp[i] [0] = false, dp[0] [j] = dp[0] [j - 2] if p[j - 1] == '*'  dp[0] [0] = true

     ```c++
     for (int i = 1; i < slen + 1; i++) {
       for (int j = 1; j < plen + 1; j++) {
         if (p[j - 1] != '*') {
           dp[i][j] = dp[i - 1][j - 1] && (s[i - 1] == p[j - 1] || p[j - 1] == '.');
         } else {
           if (s[i - 1] != p[j - 2] && p[j - 2] != '.') {
             dp[i][j] = dp[i][j - 2];
           } else { // abc与abc*
             dp[i][j] = dp[i][j - 2] || dp[i][j - 1] || dp[i - 1][j];
           }
         }
       }
     }
     // 递归写法
     bool isMatch(string s, string p) {
       if (p.length == 0) {
         return s.length == 0;
       }
       bool first_match = (s.length() > 0 && (s[0] == p[0] || p[0] == '.'));
       if (p.length() >= 2 && p[1] == '*') {
         return isMatch(s, p.substr(2, p.length() - 2)) || (first_match && isMatch(s.substr(1, s.length() - 1), p));
       } else {
         return first_match && isMatch(s.substr(1, s.length() - 1), p.substr(1, p.length() - 1));
       }
     }
     ```

9. **盛最多水的容器**：双指针，每次选择小的方向往前/往后移动 [LC](https://leetcode-cn.com/problems/container-with-most-water/)
10. **罗马数字转整数**： 一次遍历，遍历到第i时，看i + 1的数字，来判断+还是-。[LC](https://leetcode-cn.com/problems/roman-to-integer/)

11. **数字转罗马数字**：这个就把所有的罗马数字对应的数字列举出来，然后循环相减。 [LC](https://leetcode-cn.com/problems/integer-to-roman/)

    ```c++
    string res = "";
    string s[] = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
    int digits[] = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
    for (int i = 0; i < sizeof(digits) / sizeof(int); i++) {
      while(num >= digits[i]) {
        res += s[i];
        num -= digits[i];
      }
      if (num == 0) {
        break;
      }
    }
    ```

12. **最长公共前缀**：横向比较，每次取两个算出最长公共前缀，得到的结果和后面一个继续算。[LC](https://leetcode-cn.com/problems/longest-common-prefix/)

13. **三数之和**：排序+双指针 [LC](https://leetcode-cn.com/problems/3sum/)

    ```C++
    vector<vector<int>> res;
    sort(nums.begin(), nums.end());
    if (nums.size() < 3) return res;
    int n = nums.size(), third = 0, target = 0;
    for (int first = 0; first < n; first++) {
      /*先去重*/
      if (first > 0 && nums[first] == nums[first - 1]) {
        continue;
      }
      third = n - 1;
      target = -nums[first];
      for (int second = first + 1; second < n; second++) {
        /*先去重*/
        if (second > first + 1 && nums[second] == nums[second - 1]) {
          continue;
        }
        while (third > second && nums[third] + nums[second] > target) {  //第二个指针从后往前遍历
          third--;
        }
        if (third == second) /*找不到，则后面second+1已经没有意义了，直接跳出循环，开始first+1的下一轮*/
          break; 
        if (nums[third] + nums[second] == target) {
          res.push_back({nums[first], nums[second], nums[third]});
        }
      }
    }
    return res;
    // 注意四数求和亦是如此，多加一个for，另外target = -nums[first] - num[second]
    ```

    

14. **电话号码的字母组合**：当循环数不定时，就去递归/回溯吧。[LC](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

15. **删除链表倒数第N个节点**：双指针，先走N个节点，再一起走。找到删除。 注意用一个dummy节点，放到头部。[LC](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

16. **有效的括号**：栈解决，遇到左括号，就进栈，遇到右括号，就出栈，出的时候判断对错。[LC](https://leetcode-cn.com/problems/valid-parentheses/)

17. **合并两个有序链表**：头部的判断，为了不用dummy，先判断两个链表是否都为空。[LC](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

18. **括号生成**：组合问题，就上回溯，回溯就是不断加左括号，回溯，弹出左括号，左括号数量大于右括号时，加右括号，回溯，弹出右括号。[LC](https://leetcode-cn.com/problems/generate-parentheses/)

19. **合并K个有序链表**：归并排序的典型应用。[LC](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

    ```c++
    ListNode* Merge(vector<ListNode*> &lists, int left, int right) {
      if (left == right) {
        return lists[left];
      }
      if (left > right) {
        return nullptr;
      }
      int mid = left + (right - left) / 2;
      ListNode * L = Merge(lists, left, mid);
      ListNode * R = Merge(lists, mid + 1, right);
      return MergeTwoLists(L, R);
    }
    ```

    

20. **删除有序数组中的重复项**：用一个临时变量一致保存前面的不一样的元素。[LC](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

21. **实现strStr()**：KMP算法，没法，只能硬背，不要尝试理解了，太费时间。[LC](https://leetcode-cn.com/problems/implement-strstr/)

22. **两数相除**：倍增方法，注意把数都处理成负的，防止越界。[LC](https://leetcode-cn.com/problems/divide-two-integers/)

23. **旋转图像（90度**）：两种方法：先主对角交换，再左右交换；或者 找到旋转后的递推关系：dp[i] [j]---->dp[j] [n - i - 1] [LC](https://leetcode-cn.com/problems/rotate-image/)

24. **搜索旋转排序数组**：二分，注意先用nums[mid]和nums[left]相比较，确定[left, mid]和[mid+ 1, right]那个有序，再比较target和nums[left]/nums[right]进行比较 [LC](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

25. **在排序数组中查找元素的第一个和最后一个位置**: 二分，查找左边界，右边界。[LC](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

26. **有效的数独**： 注意小数独的遍历[k / 3 * 3, k / 3 * 3 + 2]，[k % 3 * 3, k % 3 * 3 + 2] [LC](https://leetcode-cn.com/problems/valid-sudoku/)

27. **外观数列**：直接写。[LC](https://leetcode-cn.com/problems/count-and-say/)

28. **缺失的第一个正数**：[LC](https://leetcode-cn.com/problems/first-missing-positive/submissions/) 原地置换，把[1, n]范围内的数对应的下标都取负。（这步之前先把<=0的变成n+1)

29. **接雨水**：考虑每个位置接多少雨水，也就是求左右两边在该位置最高高度的较小那个减去该位置高度。[LC](https://leetcode-cn.com/problems/trapping-rain-water/)

30. **通配符匹配**：二维动态规划

    ```c++
    // dp[i][j]: 表示s的前i个字符和p的前j个字符是否匹配
    if (p[j] 是小写字母) {
        dp[i][j] = dp[i - 1][j - 1] && s[i] == p[j]
    }
    if (p[j] 是 ? ) {
        dp[i][j] = dp[i - 1][j - 1]
    }
    if (p[j] 是 '*') {
        dp[i][j] = dp[i][j - 1] || dp[i - 1][j]
    }
    ```

    

31. **全排列**：回溯法[LC](https://leetcode-cn.com/problems/permutations/)

32. **字母异位词分组**：使用hash，key是排序后的字符串，value是vector<string>，存的是原始的key。

33. **Pow(x, n)**：主要考虑越界

34. **最大子序和**：一维动态规划，dp[i] = dp[i - 1] + nums[i - 1] if dp[i - 1] > 0 [LC](https://leetcode-cn.com/problems/maximum-subarray/)

35. **螺旋矩阵**：确定好上下左右四个边界，遍历时一直更新，注意跳出条件[LC](https://leetcode-cn.com/problems/spiral-matrix/)

36. **跳跃游戏**：解法1: dp[i] = dp[i - k] && nume[i - k] >= k. 解法2: 贪心 记录当每一步能到的最大值.该值要大于等于i。[LC](https://leetcode-cn.com/problems/jump-game/)

    ```C++
    int rightmost = 0;
    for (int i = 0; i < n; ++i) {
      if (i <= rightmost) {  // 前面能达到的最大值(最大下标)
        rightmost = max(rightmost, i + nums[i]);
        if (rightmost >= n - 1) {
          return true;
        }
      } else {
        return false;
      }
    }
    return false;
    ```

    

37. **合并区间**：先排序，然后插入。[LC](https://leetcode-cn.com/problems/merge-intervals/submissions/)

38. **不同路径**：解法1: 动态规划。 解法2: 数学方法，组合问题。[LC](https://leetcode-cn.com/problems/unique-paths/)

39. **加1**: 注意对9的处理。[LC](https://leetcode-cn.com/problems/plus-one/submissions/)

40. **x的平方根**：典型的二分求左边界。[LC](https://leetcode-cn.com/problems/sqrtx/)

41. **爬楼梯**：简单的dp。[LC](https://leetcode-cn.com/problems/climbing-stairs/)

42. **矩阵置零**：用第一行第一列记录该行该列是否有0，然后再单独处理首列首行。[LC](https://leetcode-cn.com/problems/set-matrix-zeroes/)

43. **颜色分类**：双指针。[LC](https://leetcode-cn.com/problems/sort-colors/)

44. **最小覆盖子串**：滑动窗口。比较难，用两个map存储，一个用于存储信息，一个用于记录窗口里的信息。[LC](https://leetcode-cn.com/problems/minimum-window-substring/)

45. **子集**：回溯法，选择本次元素，回溯，不选本次元素，回溯 [LC](https://leetcode-cn.com/problems/subsets/)

    ```c++
    // 两种回溯法
    void backtrack(vector<int> nums, vector<vector<int>> &res, vector<int> &temp, int begin) {  // 需要传入本次元素的索引，传递引用是带着之前的记忆
      if (begin == nums.size()) {  //求子集的结束条件，就是begin到最后
        res.push_back(temp);
        return;
      }
      // 本次元素选择，回溯，撤销选择
      temp.push_back(nums[begin]);
      backtrack(nums, res, temp, begin + 1);
      temp.pop_back();
      backtrack(nums, res, temp, begin + 1);
    }
    void backtrack(vector<int> nums, vector<vector<int>> &res, vector<int> &temp, int begin) {
      res.push_back(temp);
      for (int i = begin; i < nums.size(); i++) {
        temp.push_back(nums[i]);
        backtrack(nums, res, temp, i + 1); 
        temp.pop_back();
      }
    }
    ```

    

46. **单词搜索**：就是从每一个坐标，递归搜索.  回溯法。visted[i] [j] = true 回溯 false

    ```c++
    dfs(vector<vector<char>>& board, string word, vector<vector<bool>>& visited, int i, int j, int index) // i，j是坐标起始点；index是搜索到下标是index。用一个变量visited记录是否已经访问过
    // 回溯时，先判断board[i][j] == word[index]，然后判断index == word.length() - 1
    ```

47. **柱状图中最大矩形**：对于每个点，需要求出左边的第一个小于该点高度的坐标，右边第一个小于该点长度的坐标，这就需要递增栈，一个是i = 0...n-1，一个是i = n - 1...0。递增栈模板：[LC](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

    ```c++
    for (int i = 0; i < n; i++) {
      if (!st.empty() && heights[st.top() >= heights[i]]) {
        st.pop();
      }
      left[i] = st.empty() ? -1 : st.top();
      st.push(i);
    }
    ```

48. **合并两个有序数组**：从后往前遍历 [LC](https://leetcode-cn.com/problems/merge-sorted-array/)

49. 解码方法：一维动态规划，dp[i]：表示s的前i个字符编码数。[LC](https://leetcode-cn.com/problems/decode-ways/)

    ```c++
    dp[i] = dp[i - 1] if s[i - 1]  // 是合法字符
    dp[i] = dp[i] + dp[i - 2]  if s[i - 2, i - 1]  // 是合法字符
    ```

50. **二叉树的中序遍历**：递归和非递归两种写法。[LC](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

51. **编辑距离**：二维dp. [LC](https://leetcode-cn.com/problems/edit-distance/)

    ```c++
    // dp[i][j]: 表示word1的前i字母，word2的前j个字母，编辑距离
    if (word1[i - 1] == word2[j - 1])
        dp[i][j] = dp[i - 1][j - 1]
    else 
        dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1
    ```

52. **验证二叉搜索树**：递归，注意需要传递辅助信息：最小节点，最大节点。[LC](https://leetcode-cn.com/problems/validate-binary-search-tree/submissions/)

    ```c++
    isValidBSTCore(TreeNode* root, TreeNode* min_tree, TreeNode* max_tree)
    ```

53. **对称二叉树**：递归，需要辅助函数，输入是两个节点，判断这两个节点是否是对称的。[LC](https://leetcode-cn.com/problems/symmetric-tree/)

    ```c++
    bool isSymmetricCore(TreeNode* left, TreeNode* right);
    ```

54. **二叉树的层序遍历**：easy题，借助队列[LC](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

55. **二叉树的锯齿形层序遍历**：层序+双栈结合，再使用一个标识位来标识每次是左右孩子哪个先入栈。[LC](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/submissions/)

56. **二叉树的最大深度**：max(l, r) + 1，递归。 [LC](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

57. **从前序与中序遍历序列构造二叉树**：找到根，递归构造左右子树。[LC](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

58. **将有序数组转换为二叉搜索树**: 每次取中间节点，进行根节点的构建。 [LC](https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/)

59. **填充每个节点的下一个右侧节点指针**：构建辅助函数，传入左右孩子节点。[LC](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node/)

60. **杨辉三角**：easy，找到每一行元素与上一行元素的关系。[LC](https://leetcode-cn.com/problems/pascals-triangle/)

61. **买卖股票的最佳时机**：easy，记录每个元素之前的最小元素值。[LC](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

62. **买卖股票的最佳时机II，不限次数**：二维dp，其中第一维是第i天，第二维是持有/不持有。[LC](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

63. **二叉树的最大路径和**：像这种不直接的题，肯定需要辅助函数，辅助函数记录每个节点加上其一个左右孩子节点中的一个可能构成的最大的边。[LC](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/submissions/)

64. **验证回文串**：左右指针遍历即可。[LC](https://leetcode-cn.com/problems/valid-palindrome/)

65. **单词接龙**：解法1: BFS求解，利用队列存每层结果，利用hashset存字典，再用一个hashset存已经访问过的节点（必须有，否则必出现死循环）。解法2: 双向BFS，一个hashset存字典，一个hashset存正向遍历，一个hashset存反向遍历，一个hashset存已经访问过的，每层遍历，交换前两个hashset。[LC](https://leetcode-cn.com/problems/word-ladder/)

66. **最长连续序列**：一个hashset存储所有的，然后从begin开始，遍历begin，以及其所有上和下连续节点，遍历时不断删除，并同时更新最大连续值。[LC](https://leetcode-cn.com/problems/longest-consecutive-sequence/)

67. **被围绕的区域**：先对边界的O进行DFS寻找其所有的O，都修改成#，然后把矩阵都变成X，再把所有的#都修改成O。核心是DFS那块，四个方向。[LC](https://leetcode-cn.com/problems/surrounded-regions/)

68. **分割回文串**：这种分割问题，求组合的，等等，都是回溯法，这一题需要先用二维数组dp[i] [j]存储字符串中i到j是否是回文串。[LC](https://leetcode-cn.com/problems/palindrome-partitioning/)

    ```c++
    // 看下回溯的模板
    void dfs(vector<vector<string>> & res, vector<string> &temp, string s, int index) {
      if (index == s.length()) {
        res.push_back(temp);
        return;
      }
      for (int i = index; i < s.length(); i++) {
        if (dp[index][i]) {
          temp.push_back(s.substr(index, i - index + 1));
          dfs(res, temp, s, i + 1);
          temp.pop_back();
        }
      }
    }
    ```

69. **加油站**：贪心，用两个变量分别记录gas-cost的累积和total和cur，如果cur小于0，则重新置cur=0，更新res为下一个i，继续记录。[LC](https://leetcode-cn.com/problems/gas-station/submissions/)

    ```c++
    for (int i = 0; i < gas.size(); i++) {
      total += gas[i] - cost[i];
      cur += gas[i] - cost[i];
      if (cur < 0) {
        res = (i + 1) % gas.size();
        cur = 0;
      }
    }
    ```

70. **只出现一次的数字**：使用异或操作。[LC](https://leetcode-cn.com/problems/single-number/)

71. **复制带随机指针的链表**：在每个节点后面多加一个节点，都加完以后再单独处理每个节点的random节点，然后再拆分。[LC](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)

72. **单词拆分**：一维动态规划。 拿到题目如果没有思路，就想想能不能用动态规划来解决，不能再考虑回溯，BFS等方法。[LC](https://leetcode-cn.com/problems/word-break/)

    ```c++
    // d[i] : 表示前i个字符是否可以被拆分都在单词表里
    dp[i] = dp[i - k] && s[i-k+1...i] in wordDict
    ```

73. **单词拆分II**：就是求所有的可能，这种就是回溯法（DFS），确定回溯的输入参数，以及退出条件，回溯的可能步骤。[LC](https://leetcode-cn.com/problems/word-break-ii/submissions/)

74. **LRU缓存机制**：使用双向链表存储key，value。使用unordered_map存储key和节点，方便寻找。注意get操作需要把访问的节点移动到链表头，put操作需要把访问的节点移动到头，超出capacity 的，要删除尾部的，所有过程都要更新map。[LC](https://leetcode-cn.com/problems/lru-cache/)

75. **排序链表**：归并排序，每次找到链表的中点（注意奇数节点个数/偶数节点个数有点区别），把中点->next修改成nullptr。然后对这两段分别调用排序，排好序的，子集再调用合并两个链表的操作。[LC](https://leetcode-cn.com/problems/sort-list/)

76. **直线上最多的点数**：对于每一个点，统计该点与其他点的斜率，用unordered_map的key存储斜率，用value存储个数。[LC](https://leetcode-cn.com/problems/max-points-on-a-line/)

77. **基本计算器II**：[LC](https://leetcode-cn.com/problems/basic-calculator-ii/)

78. **寻找峰值**：这个二分，有点意思。[LC](https://leetcode-cn.com/problems/find-peak-element/)

79. **分数到小数**：长除法，先判断分子分母是否为0，再判断正负，再判断是否能够整除，然后循环存储，并记录余数。[LC](https://leetcode-cn.com/problems/fraction-to-recurring-decimal/submissions/)

80. **多数元素**：[LC](https://leetcode-cn.com/problems/majority-element/)

81. **逆波兰表达式求值**：利用栈。[LC](https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/)

82. **乘积最大子数组**：使用两个一维dp，注意压缩成临时状态时，变量不要叠加，要换成其他变量代替。[LC](https://leetcode-cn.com/problems/maximum-product-subarray/submissions/)

    ```c++
    max_dp[i] = max(min_dp[i - 1] * nums[i - 1], max_dp[i - 1] * nums[i- 1], nums[i - 1]);
    min_dp[i] = min(min_dp[i - 1] * nums[i - 1], max_dp[i - 1] * nums[i- 1], nums[i - 1]);
    ```

83. **最小栈**：使用两个栈。[LC](https://leetcode-cn.com/problems/min-stack/)

84. **相交链表**：先计算出两个链表长度差，然后让长的链表先走这个差，再一起走。[LC](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

85. **N皇后**：经典回溯。[LC](https://leetcode-cn.com/problems/n-queens/)

86. **阶乘后的零**：[LC](https://leetcode-cn.com/problems/factorial-trailing-zeroes/)

    ```c++
    return n == 0 ? 0 : n / 5 + trailingZeroes(n / 5);
    ```

87. **最大数**：排序+重构比较函数。[LC](https://leetcode-cn.com/problems/largest-number/)

88. **旋转数组**：三次翻转。[LC](https://leetcode-cn.com/problems/rotate-array/)

89. **颠倒二进制位**：利用左移右移，n & 1取最后一位。[LC](https://leetcode-cn.com/problems/reverse-bits/)

90. **位1的个数**：每次利用n & (n - 1)去掉右边的1 [LC](https://leetcode-cn.com/problems/number-of-1-bits/)

91. **打家劫舍**：一维dp，easy。[LC](https://leetcode-cn.com/problems/house-robber/)

    ```
    dp[i]表示抢劫到第i家，累计最大收益
    dp[i] = max(dp[i - 1], dp[i - 2] + nums[i]);  这种都可以进行状态压缩
    ```

    

92. **打家劫舍III**：后续遍历。[LC](https://leetcode-cn.com/problems/house-robber-iii/)

    ```c++
    正确的理解:  自底向上的
    rob(root)就是以root为根，所能偷到的最高金额，这里涉及到可能偷root，也可能不偷root的
    由于不能偷相邻的，如果偷root，则不能偷其左右孩子，但是可以偷其左右孩子的孩子
    max(root->val + rob(root->left->left) + rob(root->left->right) + rob(root->right->left) + rob(root->right->right),
    rob(root->left) + rob(root->right))
    使用map存储，减少遍历
    ```

    

93. **岛屿数量**：[LC](https://leetcode-cn.com/problems/number-of-islands/)

94. **快乐数**：用一个set保存每次变换结果，如果某次结果在set里，说明重复出现过。[LC](https://leetcode-cn.com/problems/happy-number/)

95. **计数质数**：筛选法。[LC](https://leetcode-cn.com/problems/count-primes/)

96. **反转链表**：[LC](https://leetcode-cn.com/problems/reverse-linked-list/)

97. **课程表**：要做某事，必须先做其他事，问是否有可能，说白了就是判断图中是否有环，拓扑排序。两种做法：一种是DFS：

    ```c++
    // dfs判断是否有环，外层就循环遍历每个节点
    bool dfs(vector<vector<int>> &graph, vector<int> &flag, int i) {
      if (flag[i] == 1) {
        return false;
      }
      if (flag[i] == -1) {
        return true;
      }
      flag[i] = 1; // 本次标记为访问过，假设有环路
      for (auto g : graph[i]) {
        if (!dfs(graph, flag, g)) {
          return false;
        }
      }
      flag[i] = -1; //这个节点被访问过，且没有环路
      return true;
    }
    ```

    一种是BFS：先构建邻接矩阵，并统计每个课程（节点）的入度。把入度为0的都加到队列中，然后遍历队列，弹出元素（入度为0的节点），以从邻接矩阵中找到以该节点为入度的其他节点，并分别将它们的入度都减去一。最后判断所有节点的入度是否为0。[LC](https://leetcode-cn.com/problems/course-schedule/)，课程表II：[LC](https://leetcode-cn.com/problems/course-schedule-ii/)

98. 实现 Trie (前缀树)：创建时，先创建26个孩子节点，并都置为nullptr。insert时，对于某个字母，先判断以其为索引的孩子节点是否存在，不存在需要创建，遍历到最后，需要置结束标识位。[LC](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)

99. 单词搜索II：[LC](https://leetcode-cn.com/problems/word-search-ii/)

100. 存在重复元素：排序/哈希表[LC](https://leetcode-cn.com/problems/contains-duplicate/)

101. 天际线：使用一个vector排序，遍历，放到multiset里（右端点放进去，左端点出来），获取最大高度，更新结果。[LC](https://leetcode-cn.com/problems/the-skyline-problem/)

102. 二叉树中所有距离为K的结点：[LC](https://leetcode-cn.com/problems/all-nodes-distance-k-in-binary-tree/)

103. 二叉搜索树中第K小的元素：[LC](https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/)

104. 回文链表：找到链表中点进行反转后半部分，再比较。[LC](https://leetcode-cn.com/problems/palindrome-linked-list/)

105. 二叉树的最近公共祖先：[LC](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

     ```c++
     //递归思路
     F(TreeNode * root, TreeNode* p, TreeNode* q) {
       if (root == p || root == q || root == nullptr) {
         return root;
       }
       left = F(root->left, p, q);
       right = F(root->right, p, q);
       if (left && right) {
         return root;
       } else if (!left && right) {
         return right;
       } else if (left && !right) {
         return left
       } else {
         return nullptr;
       }
     }
     ```

106. 删除链表中的节点：赋值为链表下一个节点的值。[LC](https://leetcode-cn.com/problems/delete-node-in-a-linked-list/)

107. 除自身以外数组的乘积：左右乘积列表。[LC](https://leetcode-cn.com/problems/product-of-array-except-self/)

108. 滑动窗口最大值：维持一个非严格单调递减队列。[LC](https://leetcode-cn.com/problems/sliding-window-maximum/)

     ```c++
     class MyDeque {
     private:
       deque<int> q;
     public:
       MyDeque() {}
       void push(int x) {
         while (!q.empty() && q.back() < x) {
           q.pop_back(); // 从后面删除
         }
         q.push_back(x);
       }
       void pop(int x) {
         if (!q.empty() && q.front() == x) {
           q.pop_front();  // 从前面弹出
         }
       }
       int getmax() {
         return q.front();
       }
     }
     ```

109. 搜索二维矩阵：右上->左下搜索，可以用二分加快搜索[LC](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)

110. 有效的字母异位词：[LC](https://leetcode-cn.com/problems/valid-anagram/)

111. 缺失的数字：求和，再减。[LC](https://leetcode-cn.com/problems/missing-number/)

112. 完全平方数：返回和为n的完全平方数的最少数量，求最少这种问题，首先想到dp。

     dp[n] = min(dp[n - i * i]). for i = 0...sqrt(n)   init dp[n] = i. [LC](https://leetcode-cn.com/problems/perfect-squares/)

113. 移动零：把0移动数组的最左边/最右边，双指针，一个遍历，一个记录0的位置。[LC]()

```c++
for (int i = 0; i < n; i++) {
  if (nums[i] == 0) {
    continue;
  } else {
    swap(nums[p_0], nums[i]);
    p_0++;
  }
}
```

114. 寻找重复数：以对应数字-1的下标的数字改成负的。[LC](https://leetcode-cn.com/problems/find-the-duplicate-number/)

115. 生命游戏：每个位置都记录附近活细胞数目，对于该位置本来就是活细胞的，用正数表示；本来是死细胞的，用负数表示。[LC](https://leetcode-cn.com/problems/game-of-life/)

116. 数据流中的中位数：使用两个优先队列，一个是大顶堆（默认的）存放一半较小的数字，一个是小顶堆，存放一半较大的数字。[LC](https://leetcode-cn.com/problems/find-median-from-data-stream/) 插入是O(nlog(n))

117. 二叉树的系列化和反序列化：序列化，就是前序遍历的递归。反序列化：dfs。这种写法不太好想，按照层序遍历写，会好点。[LC](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/)

118. 最长递增子序列：两种解法，一种一维dp[i]：表示以i为结尾的最长递增子序列。

     ```c++
     dp[i] = max(dp[j] + 1)  0 <= j < i && nums[j] < nums[i]
     ```

     一种是Nlog(N)

     ```c++
     二分思想。
     维护一个数组，每次遍历一个数时，就把该数替换数组中比该数第一个大的数，如果没有比这个数大的数，就把这个数加到最后。
     注意： 把该数替换为数组中比该数第一个大的数，其实就是一个有序数组找，二分查找
     ```

     [LC](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

119. 计算右侧小于当前元素的个数：归并排序，相比逆序对，多了一个索引的存储。[LC](https://leetcode-cn.com/problems/count-of-smaller-numbers-after-self/)

120. 数组中的逆序对：归并排序，归并的过程中，统计逆序对。[LC](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

121. 零钱兑换：动态规划，完全背包问题。[LC](https://leetcode-cn.com/problems/coin-change/)

     ```c++
     dp[i][s]表示前i个硬币，凑成金额s，所需要的最少的硬币数量
     不用第i个，以及第i个用很多个
     dp[i][s] = min(dp[i][s - k * coins[i - 1]] + k, dp[i - 1][s])   for k = 1...s/coins[i - 1]
     完全背包问题，状态方程可以退化为：
     dp[i][s] = min(dp[i][s - coins[i - 1]] + 1, dp[i - 1][s])
     解释一下：
     dp[i - 1][s - coins[i - 1]]  本身包含了dp[i - 1][s - 2 * coins[i - 1]] ...
     实际编码时，可以使用状态压缩。
     ```

122. 摆动排序II：排序 + 前后一半反转 + 穿插合并。[LC](https://leetcode-cn.com/problems/wiggle-sort-ii/)

123. 3的幂：就一直判断%3是否为0，然后除以3。[LC](https://leetcode-cn.com/problems/power-of-three/)

124. 奇偶链表：用两个指针，分别指向奇数偶数链表，交换节点。然后用一个指针记录偶数链表的头节点，最后连一下。[LC](https://leetcode-cn.com/problems/odd-even-linked-list/)

125. 矩阵中的最长递增路径：对每个点dfs回溯+状态化记忆（如果用状态化记忆，必须用返回值比较好，否则只能传递引用）。[LC](https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/) 

126. 递增的三元子序列：[LC](https://leetcode-cn.com/problems/increasing-triplet-subsequence/)

127. 扁平化嵌套列表迭代器：dfs，[LC](https://leetcode-cn.com/problems/flatten-nested-list-iterator/)

128. 反转字符串：[LC](https://leetcode-cn.com/problems/reverse-string/)

129. 前k个高频元素：先用hash map把元素和个数存起来，然后可以使用1）优先队列，2）vector+快排划分 两种方法求解。[LC](https://leetcode-cn.com/problems/top-k-frequent-elements/)

130. 两个数组的交集II：hash map [LC](https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/)

131. 两整数之和：不使用+-，求两个数异或，然后求两个数与，并转化为unsigned int，然后右移，这两个相加。[LC](https://leetcode-cn.com/problems/sum-of-two-integers/)

132. 有序矩阵中第 K 小的元素：两种解法：优先队列（基于堆的）和二分查找。[LC](https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix/)

133. 常数时间插入、删除和获取随机元素：使用一个vector用于存储数据，用一个hash map存储元素的值和索引。每次插入删除替换时都要考虑两个数据结构的更新。[LC](https://leetcode-cn.com/problems/insert-delete-getrandom-o1/)

134. 打乱数组：使用一个vector保存原来的数组，然后对每个元素，都随机和后面的元素进行交换。[LC](https://leetcode-cn.com/problems/shuffle-an-array/)

135. 字符串中的第一个唯一字符：两次遍历，用hash map存储数据。[LC](https://leetcode-cn.com/problems/first-unique-character-in-a-string/)

136. 至少有 K 个重复字符的最长子串：就dfs去寻找，对于每个字符串s，先统计其每个字母出现的次数，从头判断是否>=k，找到第一个<k的，递归其前半段和后半段。[LC](https://leetcode-cn.com/problems/longest-substring-with-at-least-k-repeating-characters/)

137. Fizz Buzz：就简单的模拟就行。[LC](https://leetcode-cn.com/problems/fizz-buzz/)

138. 四数相加II：双重for循环+hashmap 存储和和次数。[LC](https://leetcode-cn.com/problems/4sum-ii/)

139. 会议室II：排序+优先队列。[LC](https://blog.csdn.net/weixin_39722329/article/details/100641715)

140. 火星字典：拓扑排序。[LC](https://michael.blog.csdn.net/article/details/107346218)

141. 寻找名人：贪心算法。[LC](https://blog.csdn.net/qq_32424059/article/details/100550793)

142. 二叉搜索树中的顺序后继：两种解法：非递归的中序遍历 和递归的利用BST的性质 [LC](https://blog.csdn.net/qq_21201267/article/details/107131934) [LC2](https://blog.csdn.net/weixin_36094222/article/details/91309196)

143. 至多包含K个不同字符的最长子串：滑动窗口  [LC](https://blog.csdn.net/zjwreal/article/details/102070473) 




## 剑指offer里最重要的22道题-必刷

1. 0～n-1中缺失的数字：二分，注意二分初始化条件：[LC](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)

2. 二叉搜索树的第k大节点：右->中->左遍历，[LC](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

3. 数组中出现的次数：异或运算。[LC](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

4. 数组中数字出现的次数II：bit位操作，就是统计某个bit位出现的次数，再对3取模。[LC](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/)

5. 旋转数组的最小数字：套用二分查找，旋转数组的话，比较num[mid]和num[right]，没有重复的好比较，有重复的，如果相等时，只能right--，这个细节注意。[LC](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

6. 二叉树之字打印：利用两个栈模拟，左右节点入栈和层的奇偶性。[LC](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/)

7. 二叉搜索树与双向链表：中序遍历，dfs，在遍历时引入两个辅助节点，head和pre（当前节点的前一个节点，由于无法获得后续节点，所以只能记录前一个节点。遍历到根节点，更新指针指向。）[LC]([二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/))

   ```c++
   Node* treeToDoublyList(Node* root) {
           if (root == NULL) {
               return root;
           }
           dfs(root);
           pre->right = head;
           head->left = pre;
           return head;
       }
       void dfs(Node* root) {
           if (root == NULL) {
               return;
           }
           dfs(root->left); // 左
           if (pre == NULL) {
               head = root;  //找到最左边的节点了
               pre = root;
           } else {
               pre->right = root;
               root->left = pre;
           }
           pre = root;
           dfs(root->right);
       }
   ```

   

8. 字符串的排列：回溯法 + set去重。全是细节。[LC](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/)

9. 数字序列中某一位的数字: 统计每一数字的个数，以及位数，不断减去，获得可能在第哪个个数上的数字，然后获得第几个数字，进而获得该数字的第几位。 [LC](https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/)

10. 1～n 整数中 1 出现的次数：获得每一位的数字cur，其高位的数high，低位的数low，以及该数字在第几位表示成10^i，即digit，然后根据cur的值来判断。本质上就是以cur=1，可能数。[LC](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/)

11. 把数字翻译成字符串：动态规划。[LC](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

12. 礼物的最大价值：简单的二维动态规划。[LC](https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof/)

13. 最长不含重复字符的子字符串：简单的滑动窗口题目。[LC](https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

14. 丑数：记录上一个乘以2，3，5后大于当前数的数对应的索引，以及每一个丑数的值都要存下来。[LC](https://leetcode-cn.com/problems/chou-shu-lcof/submissions/)

15. 平衡二叉树：

    ```c++
    bool isBalancedCore(TreeNode* root, int & depth) { // 返回深度，并判断是否是平衡二叉树
    ```

16. 和为s的连续正数序列：[LC](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

    思路就是用双指针，开始分别指向1和2，然后判断当前sum，如果<target，right++，更新sum；如果> target，left++，然后更新sum（sum-原来的left），如果=target，则把left～right的加到结果里。

17. 滑动窗口的最大值：双端单调递减队列。[LC](https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

18. 队列的最大值：此题解法和17里的题非常类似，不过这个是加了一个队列，用于弹出用。另外还需要一个单调递减双端队列。[LC](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/)

19. n个骰子的点数：求每个点数出现的概率值，就是利用动态规划。概率题，动态规划的简单版本。[LC](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/)

    ```c++
    dp[i][j]是前i个和为j的概率
    dp[i][j] = sum(dp[i - 1][j-k]* 1 / 6);  k from 1 to n
    ```

    

20. 圆圈中最后剩下的数字：模拟思路就是从最后一轮的0开始地推这个数字是上一轮数字几，然后一直递推。[LC](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

21. 求1+2+...+n，两种解法，称之为天秀：[LC](https://leetcode-cn.com/problems/qiu-12n-lcof/)

    ```c++
    bool a[n][n + 1];
    return sizeof(a) >> 1;
    // 或者递归
    int sumNums(int n) {
      bool t = n > 1 && sumNums(n - 1);
      res += n;
      return res;
    }
    ```

22. 下一个排列：[LC](https://leetcode-cn.com/problems/next-permutation/)

    ```c++
    // 先从后往前，两两比较，找到第一个a[i] < a[i + 1]
    // 然后从最后往前找a[j] > a[i]的第一个j
    // 然后反转 i + 1到最后的元素
    ```

## 牛客网站上的Top30题

[link](https://www.nowcoder.com/activity/oj)

1. 最长回文子串：如果用动态规划的话，外层for循环是遍历长度，内层for循环遍历i。[link](https://www.nowcoder.com/practice/b4525d1d84934cf280439aeecc36f4af?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey) dp[i] [j]表示s[i] [j]是否是回文子串。

2. 链表每k个数反转：[link](https://www.nowcoder.com/practice/b49c3dc907814e9bbfa8437c251b028e?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey)

3. 三数之和：注意判断重复的细节，需要认真写好。[link](https://www.nowcoder.com/practice/345e2ed5f81d4017bbb8cc6055b0b711?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey)

4. LRU缓存：经典名题，值得多刷！[link](https://www.nowcoder.com/practice/e3769a5f49894d49b871c09cadd13a61?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey) 本题思路就是双向链表（节点）+哈希表。同时每次更新时，既考虑链表的更新和删除，又要考虑哈希表的更新和删除。

   注意，c++11没有unordered_map，需要自己添加。

5. LFU结构设计：[link](https://www.nowcoder.com/practice/93aacb4a887b46d897b00823f30bfea1?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey)  每个频率对应一个双向链表节点，同时使用两个hash表分别用来对应key-节点位置，freq-双向链表节点（这个主要用来寻找最小频率对应的节点）。

6. 两个字符串的最长公共子串：由于子串是必须连续的，所以二维dp[i] [j]：表示以字符串s1的第i个字符和字符串s2的第j个字符为结尾的最长公共子串。找到最长的，反向求结果。[link](https://www.nowcoder.com/practice/f33f5adc55f444baa0e0ca87ad8a6aac?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey)

   ```c++
   if (s[i - 1] == p[j - 1])  { dp[i][j] =  dp[i-1][j-1] + 1;}
   ```

7. 最长公共子序列：

   ```c++
   // dp[i][j]: 表示以str1的前i个字符和str2的前j个字符为结尾构成的最长公共子序列
     if (s[i - 1] == p[j - 1])
       dp[i][j] = max(dp[i][j-1], dp[i-1][j], dp[i-1][j-1] + 1)
      else 
        dp[i][j] = max(dp[i][j-1], dp[i-1][j])
   ```

8. 最长递增子序列：如果求长度，可以用动态规划以arr[i]为结尾的最长子序列长度，也可以用一个递增数组+二分。如果求序列本身，则还需要一个数组，保存以每个元素为结尾的最长递增子序列长度。有点巧妙，注意细节的优化。 [link](https://www.nowcoder.com/practice/9cf027bf54714ad889d4f30ff0ae5481?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey)

9. 合并区间： 注意区间包含的，eg：[1, 4] 和[2, 3]。 [link](https://www.nowcoder.com/practice/69f4e5b7ad284a478777cb2a17fb5e6a?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey)

10. atoi：注意细节。[link](https://www.nowcoder.com/practice/44d8c152c38f43a1b10e168018dcc13f?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey)

11. 判断是不是二叉搜索树：两种方法，递归和非递归

    ```c++
    // 递归的函数接口设计
    bool IsBst(TreeNode* root, TreeNode* min_v, TreeNode* max_v);
    // 非递归
    void inorder(TreeNode* root, vector<bool>& vet, vector<int> &v) {
      if(!root || !vet[0]) return;
      inorder(root->left, vet, v);
      if(!v.empty() && root->val < v.back()) vet[0] = false;
      v.push_back(root->val);
      inorder(root->right, vet, v);
    }
    ```

    判断是不是完全二叉树：

    ```c++
    // 层序遍历的思想，说白了就是判断某个节点，如果左/右孩子节点缺失，则打上flag，后面的所有节点都不能有孩子节点。 核心代码
    if (top->left) {
      if (IsEnd) return false;
      q.push(top->left);
    } else {
      IsEnd = true;
    }
    if (top->right) {
      if (IsEnd) return false;
      q.push(top->right);
    } else {
      IsEnd = true;
    }
    ```

12.  旋转数组寻找是否存在某个元素。 [牛客](https://www.nowcoder.com/practice/7cd13986c79d4d3a8d928d490db5d707?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey) [LC](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/) 

    注意，牛客上的答案更正确，考虑了数组不旋转的情况。

    ```
    // 解题框架仍然是二分的思想。
    1. 判断 A[mid] 和target的值
    2. 通过比较A[mid]和A[right]的值，来确定哪一半是有序的。  即要么[left, mid]是有序的，要么[mid, right]是有序的。
    3. 确定后有序后，判断 target和A[mid]的关系。再寻找二分的可能点。利用A[target]和A[right]关系。直接判断target是否在(A[mid], A[right]]范围内，或者在[A[left], A[mid])范围内。
    ```

    如果数组里有重复元素的话。[LC](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/submissions/) 只加了A[left] == A[mid]时，需要left++。

13. 滑动窗口最大值：[LC](https://www.nowcoder.com/practice/1624bc35a45c42c0bc17d17fa0cba788?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey)

14. 二叉树的遍历：三种方式，递归和非递归的。[link](https://www.nowcoder.com/practice/a9fec6c46a684ad5a3abd4e365a9d362?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey)

15. 删除链表中的重复元素：不好写，有非递归和递归两种方式：[LC](https://www.nowcoder.com/practice/71cef9f8b5564579bf7ed93fbe0b2024?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey) 

    ```c++
    ListNode* dummy = new ListNode(-1);
    auto p = dummy;
    while(p->next) {
      auto q = p->next;
      while (q && q->val == p->next->val) q = q->next;
      if (p->next->next == q) p = p->next;
      else p->next = q;
    }
    // 递归
    ListNode *deleteDuplicates(ListNode *head)
    {
      if (!head || !head->next)
        return head;
      if (head->val != head->next->val)
      {
        head->next = deleteDuplicates(head->next);
        return head;
      }
      else
      {
        int tmp = head->val;
        while (head->val==tmp)
        {
          head = head->next;
          if (!head)
            return NULL;
        }
        return deleteDuplicates(head);
      }
    }
    ```

16. **树的直径**：图的dfs。[lc](https://www.nowcoder.com/practice/a77b4f3d84bf4a7891519ffee9376df3?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey)

17. 约瑟夫环：[LC](https://blog.csdn.net/qie_wei/article/details/87915174) 

18. **01背包问题和完全背包问题**。[LC](https://www.nowcoder.com/practice/2820ea076d144b30806e72de5e5d4bbf?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey)

    ```c++
    // dp[i][v]: 前i个物品，体积为v时，最大重量. 物品不限量
      dp[i][v] = max(dp[i - 1][v], dp[i][v - vw[i - 1][0]] + vw[i - 1][1]);
    // 如果物品限量
      dp[i][v] = max(dp[i - 1][v], dp[i - 1][v - vw[i - 1][0]] + vw[i - 1][1]);
    ```

19. 通配符匹配：[LC](https://www.nowcoder.com/practice/e96f1a44d4e44d9ab6289ee080099322?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey) 

20. **KMP算法**的默写：[link](https://www.nowcoder.com/practice/bb1615c381cc4237919d1aa448083bcc?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey) 

21. 出现次数topK的问题通用解决方案：[LC](https://www.nowcoder.com/practice/fd711bdfa0e840b381d7e1b82183b3ee?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey) 

    堆排序，或者划分。

    使用堆时，一定想清楚是构造小顶堆，还是大顶堆，如果求最大的几个数，一定是构造小顶堆；否则是大顶堆。

    ```c++
    struct node {
        string s;
        int num;
        node (string _s, int _num) : s(_s), num(_num) {}
    };
    struct cmp {
        bool operator() (node& a, node& b) {  // 重载小顶堆，最后的效果，就把b想象成堆顶。
            if (a.num == b.num) {
                return a.s < b.s;
            } else {
                return a.num > b.num;
            }
        }
    };
    priority_queue<node, vector<node>, cmp> pq;  // 构造一个小顶堆.
    pq.push(node(itr->first, itr->second));
    pq.pop();
    pq.size()
    ```

22. 数组中只出现一次的数：按bit统计。 [LC](https://www.nowcoder.com/practice/5d3d74c3bf7f4e368e03096bb8857871?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey) 

23.  二叉树中的最大路径和。 注意这一题要求路径的两个端点不一定是叶子节点。注意辅助函数的定义，以及辅助函数最后返回的值的计算。我在这里吃过亏。[link](https://www.nowcoder.com/practice/da785ea0f64b442488c125b441a4ba4a?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey) 

    如果求最大直径，只需要修改辅助函数最后返回值的形式即可。

    ```c++ 
    int gain(TreeNode* root, int & max_res) // 辅助函数，返回root带来的最大增益 
    {
      // ...
      return max(0, max(left_gain, right_gain) + root->val);
      // 如果求最大直径，则return max(left_gain, right_gain) + root->val;
    }
    
    ```

24. 子数组最大乘积：注意最大的坑是，算出最新的max和min后，不能直接用，还是用之前的。[link](https://www.nowcoder.com/practice/9c158345c867466293fc413cff570356?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey) 

25. 括号生成：回溯法，引入辅助函数，引入辅助变量：左括号的数量，注意传递字符串引用时，不要带引用，这个是坑。[LC](https://www.nowcoder.com/practice/c9addb265cdf4cdd92c092c655d164ca?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey)  如果非要带引用，注意写法要修改：[LC](https://leetcode-cn.com/problems/generate-parentheses/) 

26. 最大公约数，两行代码：

    ```c++
    int gcd(int a, int b) {
      if (a % b == 0) return b;
      else return gcd(b, a % b);
    }
    ```

27. **数字字符串转化为IP地址**：就是回溯法，注意判断0数字（回溯首位遇到0，必须加分割符）。[LC](https://www.nowcoder.com/practice/ce73540d47374dbe85b3125f57727e1e?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey) 

28. **加起来和为目标值的组合**：sort + 回溯 + 去重（for循环里去重），很难想，值得多刷。。[LC](https://www.nowcoder.com/practice/75e6cd5b85ab41c6a7c43359a74e869a?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey)

29. **表达式求值**：[LC](https://www.nowcoder.com/practice/c215ba61c8b1443b996351df929dc4d4?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey)

30. N的阶乘里0的个数：[LC]()

    ```c++
    if (n == 0) {
      return 0;
    } else {
      return n / 5 + thenumberof0(n / 5);
    }
    ```

    ## 

    

    

    

    
