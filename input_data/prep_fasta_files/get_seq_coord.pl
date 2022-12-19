#!/usr/bin/perl
use strict;
use Getopt::Long;

my (@c,@cr,$h,$nd);
GetOptions('coord=s@',\@c,'help',\$h,'nodefline',\$nd);
die qq/get_seq_coord.pl -coord x1-x2[,x3-x4]... [-nodefline] fastafile [seq_id]
Get a range from a sequence:
   -coord = the range coordinates
   -nodefline = don't print 
   fastafile = fasta file with the sequence
   seq_id = sequence id for the defline
If x1>x2, returns reverse complement.
If more than one pair of coordinates are given, 
the segments that they represent are coalesced
into the final output.
/ if $h;
my $chr=$ARGV[0];
my $seq_name=$ARGV[1];
$chr =~ s/.fa//;
my $maxr=0;
my $sense;
foreach my $cset (@c) {
  my @pairs = split /,/, $cset;
  foreach (@pairs) {
    my ($left,$right)=split /\-/;
    $sense = ($left <= $right);
    my @pair=$sense?($left,$right):($right,$left);
    push @cr,\@pair;
    $maxr=$pair[1] if ($maxr<$pair[1]);
  }
}

my $seq='';
my $len=0;
while (<>){
 chomp;
 next if /^>/;
 $seq.=$_;
 $len+=length($_);
 last if ($len>$maxr);
}

$\="\n";
$,="\n";
my $n=1;
my $desc;
my $name=$seq_name?"$seq_name $chr:":$chr;
my $sequence;
foreach my $exon (@cr) {
 $desc .= ',' if $desc;
 $desc .= "$exon->[0]-$exon->[1]";
 my $eseq = substr $seq, $exon->[0] - 1, $exon->[1] - $exon->[0] + 1;
 $eseq = revcomp($eseq) unless $sense;
 $sequence .= $eseq;
 $n++;
}
$desc .= ' '.length($sequence).'nt';
print ">$name $desc" unless $nd;
print unpack "A60" x (int(length($sequence)/60)+1),$sequence;

sub revcomp{
 my $seq = $_[0];
 $seq = join ('', reverse (unpack 'A' x length($_[0]), $_[0]));
# print $seq;
 $seq =~ tr/actgACTG/tgacTGAC/;
 return $seq
}
