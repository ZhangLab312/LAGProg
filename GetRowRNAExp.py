import pandas as pd


def process_duplicates(features):
    """
    :param feature_path: {cancer}_Combined.txt
    :return:
    """

    duplicated_features = features[features.duplicated(subset='symbol')]
    duplicated_gene = duplicated_features['symbol'].values
    for gene in duplicated_gene:
        temp = features.loc[features['symbol'] == gene]
        temp_mean = temp.iloc[:, 1:].mean().values
        index = temp.index.values

        features.iloc[index, 1:] = temp_mean

    features.drop_duplicates(keep='first', inplace=True)
    features.fillna(0)
    return features


samples = pd.read_csv("Samples.csv", header=None)

RNA = pd.read_table("E:\\xiong\\Bioinfomatics\\Disease\\BRCA\\RNA.txt")
RNA['gene_id'] = RNA['gene_id'].map(lambda x: x.split('|')[0])

RNA = RNA.iloc[29:, :]
RNA = RNA.reset_index(drop=True)

barcodes = RNA.columns.to_numpy()[1:]
new_barcodes = ['symbol']

for i in barcodes:
    split_barcode = i.split('-')
    new_sample_barcode = "{}-{}-{}-{}".format(split_barcode[0], split_barcode[1], split_barcode[2],
                                              split_barcode[3][0:2])
    new_barcodes.append(new_sample_barcode)

RNA.columns = new_barcodes

RNA = RNA.loc[:, samples.to_numpy().squeeze()]
RNA = process_duplicates(RNA)

RNA.to_csv('BRCA_RNA_DE.csv', index=False)

print('')
