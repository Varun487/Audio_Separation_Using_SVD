# Taking input from the 2 audio recording

# m, fs -> m is the audio matrix, fs is sampling rate
# fs1  and fs2 should be the same
[m1, Fs1] = audioread('mixed1.wav');
disp("mixed1.wav has been read.")

[m2, Fs2] = audioread('mixed2.wav');
disp("mixed2.wav has been read.")

# N X 1
disp("size of m1");
disp(size(m1));

# N X 1
disp("size of m2");
disp(size(m2));

# N X 2
m = [m1, m2];
disp("size of m");
disp(size(m));

##########################################################
# IMPLEMENTATION OF COCKTAIL PARTY ALGORITHM #
##########################################################

# measure of how much 2 Variables Vary together
# σ(x,y)=1/n−1 * n∑i=1 (xi−¯x)(yi−¯y)
# C= σ(x,x) σ(x,y)
#    σ(y,x) σ(y,y)
# 2 X 2
cov_m = cov(m);
disp("cov_m size ");
disp(cov_m);


# 2 X 2
inv_cov_m = inv(cov_m);
disp("inv_cov_m size ");
disp(inv_cov_m);

# 2 X 2
sqrt_inv_cov_m = sqrtm(inv_cov_m);
disp("sqrt_inv_cov_m size ");
disp(sqrt_inv_cov_m);

# 2 X 1
row_wise_mean_xx = mean(m',2);
disp("row_wise_mean_xx size ");
disp(row_wise_mean_xx);

# Turn row_wise_mean_xx to a 2 X N ( Broadcasting )
mean_mat = repmat(mean(m',2), 1, size(m',2));
disp("mean_mat size ");
disp(size(mean_mat));

# (2 X 2) * (2 X N) = (2 X N)
# processed audio
p_audio = sqrt_inv_cov_m * (m' - mean_mat);
disp("size of p_audio");
disp(size(p_audio));

# 2 X N
p_audio_sq = p_audio .* p_audio;
disp("size of p_audio_sq");
disp(size(p_audio_sq));

# row wise sum
# 1 X N
sum_p_audio_sq = sum(p_audio .* p_audio,1);
disp("size of sum_p_audio_sq");
disp(size(sum_p_audio_sq));

# broadcasting the previous matrix
# 2 X N
resize_sum_p_audio_sq = repmat(sum_p_audio_sq,size(p_audio,1),1);
disp("size of resize_sum_p_audio_sq");
disp(size(resize_sum_p_audio_sq));

# [ U, Sigma, V_transpose ]
[U,s,v_T] = svd((resize_sum_p_audio_sq.*p_audio)*p_audio');

resultant = U * p_audio;
##########################################################

##########################################################
# GRAPHICAL ANALYSIS OF RESULT #
##########################################################

subplot(2,2,1); plot(m1); title('mixed audio - mic 1');
subplot(2,2,2); plot(m2); title('mixed audio - mic 2');
subplot(2,2,3); plot(resultant(1,:), 'g'); title('unmixed wave 1');
subplot(2,2,4); plot(resultant(2,:),'r'); title('unmixed wave 2');

##########################################################


# Scale down by an empiric value
output = (resultant .* 0.1)';

audiowrite('separated1.wav', output(:, 1), Fs1);
audiowrite('separated2.wav', output(:, 2), Fs1);
