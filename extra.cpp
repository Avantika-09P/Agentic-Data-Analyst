int jump(vector<int>& nums){      // {1,2,1,1,1} - 
    int farthest = 0;             // {0,1,2,3,4} 
    int currentEnd = 0;
    int jumps = 0;
    for(int i = 0; i < nums.size()-1; i++){
        farthest = max(farthest, i + nums[i]);
        if(i==currentEnd){
            jumps++;
            currentEnd = farthest;
        }
    }
    return jumps;
}

//0,1 0==0yes jumps=1 , ce=1
//1,3->3 , 1==1yes; jumps = 2; ce = 3
//3,3->3 2==3no; jumps = 2; ce = 3
//3,4->4 3==3yes->jumps = 3; ce = 4


int Profit(vector<int>& nums){
    int minPrice = nums[0];
    int maxProfit = 0;
    for(int i = 1; i<nums.size();i++){
        minPrice = min(minPrice, nums[i]);
        maxProfit = max(maxProfit, nums[i] - minPrice); //SP-CP
    }
    return maxProfit;
}

// {7,1}
//i=0:minPrice = 7, maxProfit - 0;
//i=1:minPrice = min(7, 1)=1; maxProfit = max(0,1-1)=0




int majorityElement(vector<int>& nums) {
    int candidate = 0, count = 0;
    for(int i = 0; i<nums.size();i++){
        if(count==0){
            candidate = nums[i];
        }
        if(nums[i]==candidate){
            count++;
        } 
        else{
            count--;
        }
    }
    return candidate;
}




void nextPermutation(vector<int>& nums){
    int n = nums.size(); int pivot;
    for(int i = n-1; i>=0; i--){
        if(nums[i]>nums[i+1]){
            pivot = i;
        }
        for(int j = n-1; j>=pivot; i--){
            if(nums[pivot]>nums[j]){
                swap(nums[pivot], nums[j]);
            }
        }
        for(int j = pivot; j<n; j++){
            sort(nums.begin(j), nums.end());
        }
    }
}












