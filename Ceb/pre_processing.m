function [] = PM_watershed(prob_dir, marker_dir, result_dir)

    mkdir(result_dir);

    filename = dir(fullfile(prob_dir, '*.png'))
    if length(filename) > 0
        for i  = 1:length(filename)
            prob = imread(fullfile(prob_dir, filename(i).name));
            marker = imread(fullfile(marker_dir, filename(i).name));
            prob = -mat2gray(prob);
            prob = imimposemin(prob, marker);
            save(fullfile(result_dir, [filename(i).name(1:end-4) '.m']), 'prob');

        end
    end
end
