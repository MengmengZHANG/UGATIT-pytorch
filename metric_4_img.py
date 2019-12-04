from fid import fid
from kid import kid_kid, kid_is

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-m", "--metric", dest="metric", default="all",
                      help="Set batch size to use for InceptionV3 network",
                      type=str)

    parser.add_option("--p1", "--path1", dest="path1", default=None,
                      help="Path to directory containing the real images")
    parser.add_option("--p2", "--path2", dest="path2", default=None,
                      help="Path to directory containing the generated images")
    parser.add_option("-b", "--batch-size", dest="batch_size", default=1,
                      help="Set batch size to use for InceptionV3 network",
                      type=int)
    print ('------------------------options-------------------------')
    options, _ = parser.parse_args()
    train_A_path = 'dataset/selfie2anime_64_64/trainA'
    train_B_path = 'dataset/selfie2anime_64_64/trainB'
    print ('here')
    if options.metric == 'all':
        print ('calculating is now...')
        print ('is score trainA vs output_B2A:', kid_is(options.path1, 16))
        print ('is score trainB vs output_A2B:', kid_is(options.path2, 16))
        
        print ('calculating fid now...')
        print ('fid score trainA vs output_B2A:', fid(train_A_path, options.path1, 8))
        print ('fid score trainB vs output_A2B:', fid(train_B_path, options.path2, 8))

        print ('calculating kid now...')
        print ('kid score trainA vs output_B2A:', kid_kid(train_A_path, options.path1, 16))
        print ('kid score trainB vs output_A2B:', kid_kid(train_B_path, options.path2, 16))
    
    if options.metric == 'fid':
        print ('calculating fid now...')
        print ('fid score trainA vs output_B2A:', fid(train_A_path, options.path1, 8))
        print ('fid score trainB vs output_A2B:', fid(train_B_path, options.path2, 8))
    
    if options.metric == 'is':
        print ('calculating is now...')
        print ('is score trainA vs output_B2A:', kid_is(options.path1, 16))
        print ('is score trainB vs output_A2B:', kid_is(options.path2, 16))

    if options.metric == 'kid':
        print ('calculating kid now...')
        print ('kid score trainA vs output_B2A:', kid_kid(train_A_path, options.path1, 16))
        print ('kid score trainB vs output_A2B:', kid_kid(train_B_path, options.path2, 16))
    
