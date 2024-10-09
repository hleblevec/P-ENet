#This script pulls our fork of the FINN repo, that adds support for the TySOM-3-ZU7EV board,
#rectangular nearest-neighbor upsampling and a few necessary changes to the FINN compiler.
#This script uses an anonymous repository to store the forked code during the review time, and will
# be changed to a proper git pull after the review process.

wget https://anonymous.4open.science/api/repo/finn-fork-DAF6/zip -O finn.zip
mkdir finn
unzip finn.zip -d finn
chmod -R 755 finn