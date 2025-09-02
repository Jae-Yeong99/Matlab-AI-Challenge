 clear
 clc

%% 데이터 로드
foldername = "Dataset";
imds = imageDatastore(foldername, 'IncludeSubfolders', true, "LabelSource", "foldernames");
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

%% 네트워크 불러오기 및 수정
net = resnet50
inputSize = net.Layers(1).InputSize;
numClasses = numel(categories(imdsTrain.Labels));

lgraph = layerGraph(net);  % LayerGraph 사용 (중요!)

% 기존 fully connected, softmax, classification layer 제거
lgraph = removeLayers(lgraph, {'fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000'});

% 새 출력층 추가
newLayers = [
    fullyConnectedLayer(numClasses, 'Name','new_fc', ...
        'WeightLearnRateFactor',20, 'BiasLearnRateFactor',20)
    softmaxLayer('Name','new_softmax')
    classificationLayer('Name','new_classoutput')
];

% 그래프에 새 레이어 연결
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'avg_pool', 'new_fc');

%% 데이터 증강
pixelRange = [-50 50];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandYReflection',true,...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange,...
    'RandRotation',[-30 30]);
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    "DataAugmentation", imageAugmenter, 'ColorPreprocessing','gray2rgb');
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, ...
    'ColorPreprocessing','gray2rgb');

%% 학습 옵션 설정
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 8, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 3, ...
    'Verbose', false, ...
    'Plots','training-progress');

%% 학습
netTransfer = trainNetwork(augimdsTrain, lgraph, options);

%% 저장
save("netTransferRES50.mat", "netTransfer")

%% 검증 결과 시각화
[Ypred, scores] = classify(netTransfer, augimdsValidation);
idx = randperm(numel(imdsValidation.Files), 9);
figure
for i = 1:9
    subplot(3,3,i)
    I = readimage(imdsValidation, idx(i));
    imshow(I)
    title(string(Ypred(idx(i))));
end
%% ROC Curve
YTrue = imdsValidation.Labels;
classes = categories(YTrue);
numClasses = numel(classes);
figure;
hold on;
colors = lines(numClasses);  % 색상 설정

for i = 1:numClasses
    [X,Y,T,AUC] = perfcurve(YTrue, scores(:,i), classes{i});
    plot(X,Y, 'LineWidth', 2, 'DisplayName', sprintf('%s (AUC = %.2f)', classes{i}, AUC), 'Color', colors(i,:));
end

xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');
legend('Location','Best');
grid on;
hold off;

